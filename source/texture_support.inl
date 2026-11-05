// Included inside application's anonymous namespace after the buffer helpers.
struct Texture {
    VkImage image{};
    VmaAllocation allocation{};
    VkImageView view{};
};
struct Material {
    std::string name;
    size_t albedo = 0;
    size_t specular = 0;
    size_t emissive = 0;

    bool flipV = false;
};
std::vector<Texture> textures;
std::vector<Material> materials;
VkSampler textureSampler{};

void immediate(const std::function<void(VkCommandBuffer)>& record) {
    auto& c=graphics::internal::context;
    VkCommandPool pool{};
    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pi.queueFamilyIndex=c.graphics_queue_index;
    pi.flags=VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    check(vkCreateCommandPool(c.device,&pi,nullptr,&pool),"upload pool");
    try {
        VkCommandBuffer cmd{};
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool=pool; ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount=1;
        check(vkAllocateCommandBuffers(c.device,&ai,&cmd),"upload command");
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(cmd,&bi),"begin upload");
        record(cmd);
        check(vkEndCommandBuffer(cmd),"end upload");
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount=1; si.pCommandBuffers=&cmd;
        check(vkQueueSubmit(c.graphics_queue,1,&si,VK_NULL_HANDLE),"submit upload");
        check(vkQueueWaitIdle(c.graphics_queue),"wait upload");
    } catch (...) { vkDestroyCommandPool(c.device,pool,nullptr); throw; }
    vkDestroyCommandPool(c.device,pool,nullptr);
}

Buffer deviceBuffer(const void* data, size_t bytes, VkBufferUsageFlags usage) {
    auto& c=graphics::internal::context;
    Buffer staging=createBuffer(bytes,VK_BUFFER_USAGE_TRANSFER_SRC_BIT), result{};
    try {
        upload(staging,data,bytes);
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size=bytes; bi.usage=usage|VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo ai{};
        ai.usage=VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        ai.requiredFlags=VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        check(vmaCreateBuffer(c.allocator,&bi,&ai,&result.handle,&result.allocation,nullptr),"device buffer");
        result.size=bytes;
        immediate([&](VkCommandBuffer cmd) {
            VkBufferCopy copy{0,0,bytes};
            vkCmdCopyBuffer(cmd,staging.handle,result.handle,1,&copy);
            VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask=(usage&VK_BUFFER_USAGE_INDEX_BUFFER_BIT)?VK_ACCESS_INDEX_READ_BIT:VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
            barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer=result.handle; barrier.size=bytes;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,0,0,nullptr,1,&barrier,0,nullptr);
        });
    } catch (...) { destroyBuffer(staging); destroyBuffer(result); throw; }
    destroyBuffer(staging);
    return result;
}

Mesh createMesh(std::vector<Vertex> vertices) {
    // Planar mapping on each face. Adjacent faces intentionally have UV seams.
    for(auto& v: vertices) {
        vec3 n=safeNormalize(v.normal);
        vec3 reference=std::abs(n.y)<0.9f?vec3(0,1,0):vec3(1,0,0);
        vec3 u=safeNormalize(glm::cross(reference,n));
        vec3 w=glm::cross(n,u);
        v.uv=glm::vec2(glm::dot(v.position,u),glm::dot(v.position,w))*0.45f+0.5f;
    }
    // Indexed mesh with actual deduplication of position/normal/UV tuples.
    std::vector<Vertex> unique;
    std::vector<uint32_t> indices;
    for (const auto& v: vertices) {
        auto it=std::find_if(unique.begin(),unique.end(),[&](const Vertex& a) {
            return a.position==v.position && a.normal==v.normal && a.uv==v.uv;
        });
        if(it==unique.end()) { indices.push_back(uint32_t(unique.size())); unique.push_back(v); }
        else indices.push_back(uint32_t(it-unique.begin()));
    }
    Mesh m{}; m.count=uint32_t(indices.size());
    m.vertices=deviceBuffer(unique.data(),unique.size()*sizeof(Vertex),VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    try { m.indices=deviceBuffer(indices.data(),indices.size()*sizeof(uint32_t),VK_BUFFER_USAGE_INDEX_BUFFER_BIT); }
    catch(...) { destroyBuffer(m.vertices); throw; }
    return m;
}

size_t loadTexture(const std::string& filename, bool srgb) {
    auto& c=graphics::internal::context;
    const auto path=std::filesystem::path(TEXTURE_DIR)/filename;
    // Decode from memory so Windows Unicode paths also work.
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if(!file) throw std::runtime_error("Missing texture: "+filename);
    const auto length=file.tellg();
    if(length<=0 || length>std::numeric_limits<int>::max()) throw std::runtime_error("Invalid texture file");
    std::vector<unsigned char> encoded(static_cast<size_t>(length));
    file.seekg(0); file.read(reinterpret_cast<char*>(encoded.data()),length);
    int width=0,height=0,channels=0;
    stbi_uc* pixels=stbi_load_from_memory(encoded.data(),int(encoded.size()),&width,&height,&channels,4);
    if(!pixels) throw std::runtime_error("Cannot decode texture: "+filename);
    Buffer staging{}; Texture t{};
    try {
        const size_t bytes=size_t(width)*size_t(height)*4;
        staging=createBuffer(bytes,VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        upload(staging,pixels,bytes);
        stbi_image_free(pixels); pixels=nullptr;
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType=VK_IMAGE_TYPE_2D;
        ii.format=srgb?VK_FORMAT_R8G8B8A8_SRGB:VK_FORMAT_R8G8B8A8_UNORM;
        ii.extent={uint32_t(width),uint32_t(height),1};
        ii.mipLevels=1; ii.arrayLayers=1; ii.samples=VK_SAMPLE_COUNT_1_BIT;
        ii.tiling=VK_IMAGE_TILING_OPTIMAL;
        ii.usage=VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_SAMPLED_BIT;
        ii.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo ai{};
        ai.usage=VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        ai.requiredFlags=VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        check(vmaCreateImage(c.allocator,&ii,&ai,&t.image,&t.allocation,nullptr),"texture image");
        immediate([&](VkCommandBuffer cmd) {
            VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            b.image=t.image; b.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
            b.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&b);
            VkBufferImageCopy copy{};
            copy.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; copy.imageExtent=ii.extent;
            vkCmdCopyBufferToImage(cmd,staging.handle,t.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&copy);
            b.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,0,0,nullptr,0,nullptr,1,&b);
        });
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image=t.image; vi.viewType=VK_IMAGE_VIEW_TYPE_2D; vi.format=ii.format;
        vi.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
        check(vkCreateImageView(c.device,&vi,nullptr,&t.view),"texture view");
        textures.push_back(t);
    } catch(...) {
        stbi_image_free(pixels); destroyBuffer(staging);
        if(t.view) vkDestroyImageView(c.device,t.view,nullptr);
        if(t.image) vmaDestroyImage(c.allocator,t.image,t.allocation);
        throw;
    }
    destroyBuffer(staging); return textures.size()-1;
}

void loadMaterials() {
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter=si.minFilter=VK_FILTER_LINEAR;
    si.mipmapMode=VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU=si.addressModeV=si.addressModeW=VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.maxLod=0;
    check(vkCreateSampler(graphics::internal::context.device,&si,nullptr,&textureSampler),"texture sampler");
    const auto black=loadTexture("black.png",false);
    const auto tiles=loadTexture("tiles.png",true);
    const auto spec=loadTexture("tiles_specular.png",false);
    const auto glow=loadTexture("glow.png",true);
    const auto emit=loadTexture("glow_emissive.png",true);
    materials.push_back({"Tiles (specular)",tiles,spec,black});
    materials.push_back({"Neon (emissive)",glow,spec,emit});
    materials.push_back({"Tiles (no extra maps)",tiles,black,black});
    for(const char* name: {"mai.png","portrait.png"}) {
        if(std::filesystem::exists(std::filesystem::path(TEXTURE_DIR)/name))
            materials.push_back({name,loadTexture(name,true),black,black});
    }
}

void destroyTextures() {
    auto& c=graphics::internal::context;
    for(auto& t:textures) {
        vkDestroyImageView(c.device,t.view,nullptr);
        vmaDestroyImage(c.allocator,t.image,t.allocation);
    }
    textures.clear(); materials.clear();
    if(textureSampler) vkDestroySampler(c.device,textureSampler,nullptr);
    textureSampler=VK_NULL_HANDLE;
}
