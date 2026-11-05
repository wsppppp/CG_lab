// One descriptor set per material, all sharing the same dynamic model UBO.
void updateModelDescriptors() {
    for(auto set:materialSets) {
        VkDescriptorBufferInfo bi{modelBuffer.handle,0,sizeof(ModelGpu)};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet=set; w.dstBinding=0; w.descriptorCount=1;
        w.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC; w.pBufferInfo=&bi;
        vkUpdateDescriptorSets(graphics::internal::context.device,1,&w,0,nullptr);
    }
}
void ensureModelCapacity(size_t slots) {
    if(slots > std::numeric_limits<uint32_t>::max()/modelStride)
        throw std::runtime_error("Dynamic UBO offset limit exceeded");
    const VkDeviceSize bytes=modelStride*slots;
    if(bytes<=modelBuffer.size) return;
    Buffer replacement=createBuffer(bytes,VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    destroyBuffer(modelBuffer); modelBuffer=replacement;
    updateModelDescriptors();
}
void writeModel(uint32_t slot,const ModelGpu& gpu) {
    const VkDeviceSize offset=modelStride*slot;
    if(offset+sizeof(gpu)>modelBuffer.size) throw std::runtime_error("Model UBO overflow");
    std::memcpy(static_cast<char*>(modelBuffer.mapped)+offset,&gpu,sizeof(gpu));
    check(vmaFlushAllocation(graphics::internal::context.allocator,modelBuffer.allocation,offset,sizeof(gpu)),"flush model");
}
void bindMaterial(VkCommandBuffer cmd,int material,uint32_t slot) {
    const uint32_t offset=uint32_t(modelStride*slot);
    vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,pipelineLayout,
        1,1,&materialSets.at(material),1,&offset);
}
void createDescriptors() {
    const auto device=graphics::internal::context.device;
    VkDescriptorSetLayoutBinding sb[3]{};
    for(uint32_t i=0;i<3;++i) {
        sb[i].binding=i; sb[i].descriptorCount=1;
        sb[i].descriptorType=i?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        sb[i].stageFlags=i?VK_SHADER_STAGE_FRAGMENT_BIT:VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount=3; li.pBindings=sb;
    check(vkCreateDescriptorSetLayout(device,&li,nullptr,&sceneLayout),"scene layout");
    VkDescriptorSetLayoutBinding mb[4]{};
    for(uint32_t i=0;i<4;++i) {
        mb[i].binding=i; mb[i].descriptorCount=1;
        mb[i].descriptorType=i?VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        mb[i].stageFlags=i?VK_SHADER_STAGE_FRAGMENT_BIT:VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    li.bindingCount=4; li.pBindings=mb;
    check(vkCreateDescriptorSetLayout(device,&li,nullptr,&modelLayout),"material layout");
    const uint32_t count=uint32_t(materials.size());
    const VkDescriptorPoolSize sizes[]={
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,2},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,count}, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,count*3}};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.maxSets=count+1; pi.poolSizeCount=4; pi.pPoolSizes=sizes;
    check(vkCreateDescriptorPool(device,&pi,nullptr,&descriptorPool),"descriptor pool");
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool=descriptorPool; ai.descriptorSetCount=1; ai.pSetLayouts=&sceneLayout;
    check(vkAllocateDescriptorSets(device,&ai,&sceneSet),"scene set");
    std::vector<VkDescriptorSetLayout> layouts(count,modelLayout);
    materialSets.resize(count);
    ai.descriptorSetCount=count; ai.pSetLayouts=layouts.data();
    check(vkAllocateDescriptorSets(device,&ai,materialSets.data()),"material sets");
    writeDescriptor(sceneSet,0,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,sceneBuffer);
    writeDescriptor(sceneSet,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,pointBuffer);
    writeDescriptor(sceneSet,2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,spotBuffer);
    updateModelDescriptors();
    for(uint32_t i=0;i<count;++i) {
        const size_t ids[]={materials[i].albedo,materials[i].specular,materials[i].emissive};
        for(uint32_t j=0;j<3;++j) {
            VkDescriptorImageInfo image{textureSampler,textures[ids[j]].view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet=materialSets[i]; w.dstBinding=j+1; w.descriptorCount=1;
            w.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo=&image;
            vkUpdateDescriptorSets(device,1,&w,0,nullptr);
        }
    }
}
