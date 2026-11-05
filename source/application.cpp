#include "application.hpp"

#include <imgui.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <filesystem>
#include <functional>
#include <limits>
#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb_image.h"

namespace application {
    namespace {

        using glm::mat4;
        using glm::vec3;
        using glm::vec4;

        constexpr float PI = 3.14159265358979323846f;
        constexpr uint32_t MODEL_COUNT = 3;

        constexpr ImGuiSliderFlags CLAMP = ImGuiSliderFlags_AlwaysClamp;

        struct Vertex {
            vec3 position;
            vec3 normal;
            glm::vec2 uv{};
        };

        struct Buffer {
            VkBuffer handle = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            void* mapped = nullptr;
            VkDeviceSize size = 0;
        };

        struct Mesh {
            Buffer vertices;
            Buffer indices;
            uint32_t count = 0;
        };

        // vec4-sized fields keep CPU and GLSL layouts identical.
        struct alignas(16) SceneGpu {
            mat4 viewProjection{ 1.0f };
            vec4 cameraPosition{};
            vec4 ambient{};
            vec4 sunDirection{};
            vec4 sunColorIntensity{};
            glm::uvec4 counts{};
        };

        struct alignas(16) ModelGpu {
            mat4 model{ 1.0f };
            mat4 normalMatrix{ 1.0f };
            vec4 albedo{ 1.0f };
            vec4 specularShininess{ 1.0f };
            vec4 options{};
        };

        struct alignas(16) PointGpu {
            vec4 position{ 0.0f, 3.0f, 2.0f, 1.0f };
            vec4 colorIntensity{ 1.0f, 0.7f, 0.3f, 25.0f };
        };

        struct alignas(16) SpotGpu {
            vec4 position{ 3.0f, 4.0f, 3.0f, 1.0f };
            vec4 directionCosOuter{ -1.0f, -1.0f, -1.0f, 0.0f };
            vec4 colorIntensity{ 1.0f, 0.95f, 0.75f, 60.0f };
            vec4 cone{};
        };

        struct Spot {
            SpotGpu gpu{};
            float innerAngle = 18.0f;
            float outerAngle = 30.0f;
        };

        struct alignas(16) Push {
            glm::ivec4 flags{};
            vec4 params{};
        };

        static_assert(sizeof(Vertex) == 32);
        static_assert(sizeof(SceneGpu) == 144);
        static_assert(sizeof(ModelGpu) == 176);
        static_assert(sizeof(PointGpu) == 32);
        static_assert(sizeof(SpotGpu) == 64);
        static_assert(sizeof(Push) == 32);

        static_assert(offsetof(SceneGpu, cameraPosition) == 64);
        static_assert(offsetof(SceneGpu, counts) == 128);
        static_assert(offsetof(ModelGpu, albedo) == 128);
        static_assert(offsetof(ModelGpu, options) == 160);
        static_assert(offsetof(Push, flags) == 0);

        struct Model {
            vec3 position{ 0.0f };
            vec3 rotation{ 0.0f };
            vec3 scale{ 1.0f };

            vec3 albedo{ 1.0f };
            vec3 specular{ 0.8f };
            float shininess = 64.0f;

            int material = 0;
            bool effect = false;
            float uvScale = 1.0f;
            float effectSpeed = 1.0f;
            float effectStrength = 0.6f;
            float emission = 1.0f;
            bool visible = true;
        };

        struct Camera {
            vec3 position{ 0.0f, 2.0f, 10.0f };

            // Pitch, yaw, roll in degrees.
            vec3 angles{ -12.0f, -90.0f, 0.0f };

            float fov = 55.0f;
            float nearPlane = 0.1f;
            float farPlane = 200.0f;
            float orthoHeight = 9.0f;
            float speed = 5.0f;

            bool perspective = true;
        };

        Camera camera;
        std::array<Model, MODEL_COUNT> models;

        std::vector<PointGpu> points;
        std::vector<Spot> spots;

        vec3 ambientColor{ 1.0f };
        float ambientIntensity = 0.08f;

        vec3 sunDirection{ -0.5f, -1.0f, -0.3f };
        vec3 sunColor{ 1.0f };
        float sunIntensity = 0.7f;
        bool sunEnabled = true;

        bool showMarkers = true;
        bool showCones = true;
        float markerSize = 0.16f;
        float coneLength = 3.0f;

        bool animateModels = false;
        float animationSpeed = 1.0f;
        float orbitRadius = 0.7f;
        float orbitHeight = 0.35f;
        float orbitFrequency = 1.0f;
        float rotationSpeed = 30.0f;

        double lastTime = -1.0;
        double animationTime = 0.0;

        Mesh dodecahedron;
        Mesh floorMesh;
        Mesh cubeMesh;
        Mesh coneMesh;

        Buffer sceneBuffer;
        Buffer pointBuffer;
        Buffer spotBuffer;
        Buffer modelBuffer;
        VkDeviceSize modelStride = 0;
        uint32_t drawSlot = MODEL_COUNT;
        float textureTime = 0;
        bool animateTextures = true;

        VkDescriptorSetLayout sceneLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout modelLayout = VK_NULL_HANDLE;
        VkDescriptorPool descriptorPool = VK_NULL_HANDLE;

        VkDescriptorSet sceneSet = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> materialSets;

        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkPipeline trianglePipeline = VK_NULL_HANDLE;
        VkPipeline linePipeline = VK_NULL_HANDLE;

        VkDeviceSize maxStorageRange = 0;

        void check(VkResult result, const char* operation) {
            if (result != VK_SUCCESS) {
                throw std::runtime_error(
                    std::string(operation) + ": VkResult = " +
                    std::to_string(static_cast<int>(result)));
            }
        }

        vec3 safeNormalize(vec3 value, vec3 fallback = vec3(0.0f, -1.0f, 0.0f)) {
            const float squaredLength = glm::dot(value, value);
            if (squaredLength < 1e-12f) {
                return fallback;
            }
            return value / std::sqrt(squaredLength);
        }

        void destroyBuffer(Buffer& buffer) {
            if (buffer.handle) {
                vmaDestroyBuffer(
                    graphics::internal::context.allocator,
                    buffer.handle,
                    buffer.allocation);
            }
            buffer = {};
        }

        Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage) {
            auto& context = graphics::internal::context;

            Buffer result{};

            VkBufferCreateInfo bufferInfo{
                VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO
            };
            bufferInfo.size = size;
            bufferInfo.usage = usage;
            bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo allocationInfo{};
            allocationInfo.flags =
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
            allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;

            VmaAllocationInfo actualInfo{};

            check(vmaCreateBuffer(
                context.allocator,
                &bufferInfo,
                &allocationInfo,
                &result.handle,
                &result.allocation,
                &actualInfo),
                "vmaCreateBuffer");

            result.size = size;
            result.mapped = actualInfo.pMappedData;

            if (!result.mapped) {
                destroyBuffer(result);
                throw std::runtime_error("Buffer is not host-mapped");
            }

            return result;
        }

        void upload(Buffer& buffer, const void* data, size_t bytes) {
            if (bytes == 0) {
                return;
            }

            if (bytes > buffer.size) {
                throw std::runtime_error("Buffer upload exceeds allocation");
            }

            std::memcpy(buffer.mapped, data, bytes);

            check(vmaFlushAllocation(
                graphics::internal::context.allocator,
                buffer.allocation,
                0,
                static_cast<VkDeviceSize>(bytes)),
                "vmaFlushAllocation");
        }

#include "texture_support.inl"
        std::vector<Vertex> createDodecahedron() {
            constexpr float phi = 1.6180339887498948482f;
            constexpr float invPhi = 0.6180339887498948482f;

            const std::array<vec3, 20> vertices = { {
                {-1, -1, -1}, {-1, -1,  1},
                {-1,  1, -1}, {-1,  1,  1},
                { 1, -1, -1}, { 1, -1,  1},
                { 1,  1, -1}, { 1,  1,  1},

                {0, -invPhi, -phi}, {0, -invPhi, phi},
                {0,  invPhi, -phi}, {0,  invPhi, phi},

                {-invPhi, -phi, 0}, {-invPhi, phi, 0},
                { invPhi, -phi, 0}, { invPhi, phi, 0},

                {-phi, 0, -invPhi}, {phi, 0, -invPhi},
                {-phi, 0,  invPhi}, {phi, 0,  invPhi}
            } };

            const std::array<std::array<int, 5>, 12> faces = { {
                {{18, 16,  0, 12,  1}},
                {{10,  8,  0, 16,  2}},
                {{14, 12,  0,  8,  4}},
                {{ 3, 18,  1,  9, 11}},
                {{ 5,  9,  1, 12, 14}},
                {{ 3, 13,  2, 16, 18}},
                {{ 6, 10,  2, 13, 15}},
                {{15, 13,  3, 11,  7}},
                {{ 5, 14,  4, 17, 19}},
                {{ 6, 17,  4,  8, 10}},
                {{11,  9,  5, 19,  7}},
                {{19, 17,  6, 15,  7}}
            } };

            std::vector<Vertex> result;
            result.reserve(108);

            for (const auto& face : faces) {
                const vec3 a = vertices[face[0]];
                const vec3 b = vertices[face[1]];
                const vec3 c = vertices[face[2]];

                vec3 normal = safeNormalize(glm::cross(b - a, c - a));

                const bool reverse = glm::dot(normal, a) < 0.0f;
                if (reverse) {
                    normal = -normal;
                }

                // Duplicate vertices across faces to preserve flat normals.
                for (int i = 1; i < 4; ++i) {
                    vec3 p1 = vertices[face[i]];
                    vec3 p2 = vertices[face[i + 1]];

                    if (reverse) {
                        std::swap(p1, p2);
                    }

                    result.push_back({ a, normal });
                    result.push_back({ p1, normal });
                    result.push_back({ p2, normal });
                }
            }

            return result;
        }

        std::vector<Vertex> createFloor() {
            const vec3 n(0.0f, 1.0f, 0.0f);

            return {
                {{-0.5f, 0.0f, -0.5f}, n},
                {{-0.5f, 0.0f,  0.5f}, n},
                {{ 0.5f, 0.0f,  0.5f}, n},

                {{-0.5f, 0.0f, -0.5f}, n},
                {{ 0.5f, 0.0f,  0.5f}, n},
                {{ 0.5f, 0.0f, -0.5f}, n}
            };
        }

        std::vector<Vertex> createCube() {
            std::vector<Vertex> result;

            for (vec3 n : {
                vec3(1, 0, 0), vec3(-1, 0, 0),
                    vec3(0, 1, 0), vec3(0, -1, 0),
                    vec3(0, 0, 1), vec3(0, 0, -1)
            }) {
                const vec3 reference =
                    std::abs(n.y) < 0.9f ? vec3(0, 1, 0) : vec3(1, 0, 0);

                const vec3 u = safeNormalize(glm::cross(reference, n)) * 0.5f;
                const vec3 v = glm::cross(n, u);
                const vec3 center = n * 0.5f;

                const vec3 a = center - u - v;
                const vec3 b = center + u - v;
                const vec3 c = center + u + v;
                const vec3 d = center - u + v;

                for (vec3 p : {a, b, c, a, c, d}) {
                    result.push_back({ p, n });
                }
            }

            return result;
        }

        std::vector<Vertex> createConeLines() {
            // Unit cone: apex at origin, axis along -Z, radius 1 at z = -1.
            constexpr int segments = 48;
            std::vector<Vertex> result;

            auto addLine = [&](vec3 a, vec3 b) {
                result.push_back({ a, vec3(0, 1, 0) });
                result.push_back({ b, vec3(0, 1, 0) });
                };

            for (int i = 0; i < segments; ++i) {
                const float a = 2.0f * PI * float(i) / float(segments);
                const float b = 2.0f * PI * float(i + 1) / float(segments);

                const vec3 p0(std::cos(a), std::sin(a), -1.0f);
                const vec3 p1(std::cos(b), std::sin(b), -1.0f);

                addLine(p0, p1);

                if (i % 12 == 0) {
                    addLine(vec3(0.0f), p0);
                }
            }

            addLine(vec3(0.0f), vec3(0, 0, -1.2f));
            return result;
        }

        void writeDescriptor(
            VkDescriptorSet set,
            uint32_t binding,
            VkDescriptorType type,
            const Buffer& buffer) {

            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = buffer.handle;
            bufferInfo.range = buffer.size;

            VkWriteDescriptorSet write{
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET
            };

            write.dstSet = set;
            write.dstBinding = binding;
            write.descriptorCount = 1;
            write.descriptorType = type;
            write.pBufferInfo = &bufferInfo;

            vkUpdateDescriptorSets(
                graphics::internal::context.device,
                1, &write, 0, nullptr);
        }

#include "material_descriptors.inl"
        template<class T>
        void uploadLights(
            Buffer& buffer,
            uint32_t binding,
            const std::vector<T>& data) {

            if (data.size() > maxStorageRange / sizeof(T)) {
                throw std::runtime_error(
                    "Light array exceeds maxStorageBufferRange");
            }

            const VkDeviceSize required = std::max<VkDeviceSize>(
                sizeof(T),
                static_cast<VkDeviceSize>(data.size()) * sizeof(T));

            if (required > buffer.size) {
                const VkDeviceSize newSize = std::max(
                    required,
                    std::min(maxStorageRange, buffer.size * 2));

                Buffer replacement = createBuffer(
                    newSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

                // prepare() already waited for the previous frame.
                // This frame has not bound this descriptor set yet.
                destroyBuffer(buffer);
                buffer = replacement;

                writeDescriptor(
                    sceneSet, binding,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, buffer);
            }

            upload(buffer, data.data(), data.size() * sizeof(T));
        }

        VkShaderModule loadShader(const char* path) {
            std::ifstream file(path, std::ios::binary | std::ios::ate);

            if (!file) {
                throw std::runtime_error(
                    std::string("Cannot open shader: ") + path);
            }

            const std::streamsize size =
                static_cast<std::streamsize>(file.tellg());

            if (size <= 0 || size % 4 != 0) {
                throw std::runtime_error("Invalid SPIR-V size");
            }

            std::vector<uint32_t> code(static_cast<size_t>(size) / 4);

            file.seekg(0);
            if (!file.read(reinterpret_cast<char*>(code.data()), size)) {
                throw std::runtime_error("Cannot read SPIR-V");
            }

            VkShaderModuleCreateInfo info{
                VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
            };

            info.codeSize = static_cast<size_t>(size);
            info.pCode = code.data();

            VkShaderModule result = VK_NULL_HANDLE;

            check(vkCreateShaderModule(
                graphics::internal::context.device,
                &info, nullptr, &result),
                "vkCreateShaderModule");

            return result;
        }

        VkPipeline createPipeline(
            VkShaderModule vertexShader,
            VkShaderModule fragmentShader,
            VkPrimitiveTopology topology) {

            auto& context = graphics::internal::context;

            VkPipelineShaderStageCreateInfo stages[2]{};

            stages[0].sType =
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vertexShader;
            stages[0].pName = "main";

            stages[1].sType =
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = fragmentShader;
            stages[1].pName = "main";

            VkVertexInputBindingDescription binding{};
            binding.binding = 0;
            binding.stride = sizeof(Vertex);
            binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

            const VkVertexInputAttributeDescription attributes[] = {
                {
                    0, 0, VK_FORMAT_R32G32B32_SFLOAT,
                    static_cast<uint32_t>(offsetof(Vertex, position))
                },
                {
                    1, 0, VK_FORMAT_R32G32B32_SFLOAT,
                    static_cast<uint32_t>(offsetof(Vertex, normal))
                },
                {2, 0, VK_FORMAT_R32G32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, uv))}
            };

            VkPipelineVertexInputStateCreateInfo vertexInput{
                VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO
            };

            vertexInput.vertexBindingDescriptionCount = 1;
            vertexInput.pVertexBindingDescriptions = &binding;
            vertexInput.vertexAttributeDescriptionCount = 3;
            vertexInput.pVertexAttributeDescriptions = attributes;

            VkPipelineInputAssemblyStateCreateInfo assembly{
                VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO
            };
            assembly.topology = topology;

            VkPipelineViewportStateCreateInfo viewport{
                VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO
            };
            viewport.viewportCount = 1;
            viewport.scissorCount = 1;

            VkPipelineRasterizationStateCreateInfo rasterization{
                VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO
            };

            rasterization.polygonMode = VK_POLYGON_MODE_FILL;
            rasterization.cullMode = VK_CULL_MODE_NONE;
            rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            rasterization.lineWidth = 1.0f;

            VkPipelineMultisampleStateCreateInfo multisampling{
                VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO
            };
            multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineDepthStencilStateCreateInfo depth{
                VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO
            };

            depth.depthTestEnable = VK_TRUE;
            depth.depthWriteEnable =
                topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
                ? VK_TRUE : VK_FALSE;
            depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

            VkPipelineColorBlendAttachmentState blendAttachment{};
            blendAttachment.colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT |
                VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT |
                VK_COLOR_COMPONENT_A_BIT;

            VkPipelineColorBlendStateCreateInfo blend{
                VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO
            };
            blend.attachmentCount = 1;
            blend.pAttachments = &blendAttachment;

            const VkDynamicState dynamicStates[] = {
                VK_DYNAMIC_STATE_VIEWPORT,
                VK_DYNAMIC_STATE_SCISSOR
            };

            VkPipelineDynamicStateCreateInfo dynamic{
                VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO
            };
            dynamic.dynamicStateCount = 2;
            dynamic.pDynamicStates = dynamicStates;

            VkGraphicsPipelineCreateInfo info{
                VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO
            };

            info.stageCount = 2;
            info.pStages = stages;
            info.pVertexInputState = &vertexInput;
            info.pInputAssemblyState = &assembly;
            info.pViewportState = &viewport;
            info.pRasterizationState = &rasterization;
            info.pMultisampleState = &multisampling;
            info.pDepthStencilState = &depth;
            info.pColorBlendState = &blend;
            info.pDynamicState = &dynamic;
            info.layout = pipelineLayout;
            info.renderPass = context.render_pass;
            info.subpass = 0;

            VkPipeline result = VK_NULL_HANDLE;

            check(vkCreateGraphicsPipelines(
                context.device, VK_NULL_HANDLE, 1,
                &info, nullptr, &result),
                "vkCreateGraphicsPipelines");

            return result;
        }

        void createPipelines() {
            const auto device = graphics::internal::context.device;

            const VkDescriptorSetLayout layouts[] = {
                sceneLayout, modelLayout
            };

            VkPushConstantRange pushRange{};
            pushRange.stageFlags =
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            pushRange.size = sizeof(Push);

            VkPipelineLayoutCreateInfo layoutInfo{
                VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
            };

            layoutInfo.setLayoutCount = 2;
            layoutInfo.pSetLayouts = layouts;
            layoutInfo.pushConstantRangeCount = 1;
            layoutInfo.pPushConstantRanges = &pushRange;

            check(vkCreatePipelineLayout(
                device, &layoutInfo, nullptr, &pipelineLayout),
                "vkCreatePipelineLayout");

            VkShaderModule vertexShader = VK_NULL_HANDLE;
            VkShaderModule fragmentShader = VK_NULL_HANDLE;

            try {
                vertexShader = loadShader(SHADER_DIR "/basic.vert.spv");
                fragmentShader = loadShader(SHADER_DIR "/basic.frag.spv");

                trianglePipeline = createPipeline(
                    vertexShader, fragmentShader,
                    VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

                linePipeline = createPipeline(
                    vertexShader, fragmentShader,
                    VK_PRIMITIVE_TOPOLOGY_LINE_LIST);
            }
            catch (...) {
                if (fragmentShader) {
                    vkDestroyShaderModule(device, fragmentShader, nullptr);
                }
                if (vertexShader) {
                    vkDestroyShaderModule(device, vertexShader, nullptr);
                }
                throw;
            }

            vkDestroyShaderModule(device, fragmentShader, nullptr);
            vkDestroyShaderModule(device, vertexShader, nullptr);
        }

        void cameraBasis(vec3& forward, vec3& right, vec3& up) {
            const float pitch = glm::radians(camera.angles.x);
            const float yaw = glm::radians(camera.angles.y);
            const float roll = glm::radians(camera.angles.z);

            forward = safeNormalize(vec3(
                std::cos(pitch) * std::cos(yaw),
                std::sin(pitch),
                std::cos(pitch) * std::sin(yaw)),
                vec3(0, 0, -1));

            right = safeNormalize(
                glm::cross(forward, vec3(0, 1, 0)),
                vec3(1, 0, 0));

            up = safeNormalize(glm::cross(right, forward), vec3(0, 1, 0));

            right = right * std::cos(roll) + up * std::sin(roll);
            up = safeNormalize(glm::cross(right, forward), vec3(0, 1, 0));
        }

        void updateCameraInput(float deltaTime) {
            ImGuiIO& io = ImGui::GetIO();

            if (!io.WantCaptureMouse) {
                camera.fov = std::clamp(
                    camera.fov - io.MouseWheel * 2.0f, 15.0f, 100.0f);

                if (ImGui::IsMouseDown(ImGuiMouseButton_Right) &&
                    !io.WantTextInput) {
                    camera.angles.y += io.MouseDelta.x * 0.18f;
                    camera.angles.x -= io.MouseDelta.y * 0.18f;
                    camera.angles.x = std::clamp(
                        camera.angles.x, -89.0f, 89.0f);

                    if (!io.WantCaptureKeyboard) {
                        vec3 forward, right, up;
                        cameraBasis(forward, right, up);

                        vec3 movement(0.0f);

                        if (ImGui::IsKeyDown(ImGuiKey_W)) movement += forward;
                        if (ImGui::IsKeyDown(ImGuiKey_S)) movement -= forward;
                        if (ImGui::IsKeyDown(ImGuiKey_D)) movement += right;
                        if (ImGui::IsKeyDown(ImGuiKey_A)) movement -= right;
                        if (ImGui::IsKeyDown(ImGuiKey_E)) movement += up;
                        if (ImGui::IsKeyDown(ImGuiKey_Q)) movement -= up;

                        if (glm::dot(movement, movement) > 0.0f) {
                            const float multiplier =
                                ImGui::IsKeyDown(ImGuiKey_LeftShift)
                                ? 3.0f : 1.0f;

                            camera.position +=
                                safeNormalize(movement) *
                                camera.speed * multiplier * deltaTime;
                        }
                    }
                }
            }

            camera.angles.y = std::remainder(camera.angles.y, 360.0f);
        }

        void drawCameraUI() {
            ImGui::Checkbox("Perspective", &camera.perspective);

            ImGui::DragFloat3(
                "Camera position", glm::value_ptr(camera.position), 0.05f);

            ImGui::SliderFloat(
                "Pitch", &camera.angles.x, -89.0f, 89.0f, "%.1f deg", CLAMP);
            ImGui::SliderFloat(
                "Yaw", &camera.angles.y, -180.0f, 180.0f, "%.1f deg", CLAMP);
            ImGui::SliderFloat(
                "Roll", &camera.angles.z, -180.0f, 180.0f, "%.1f deg", CLAMP);

            ImGui::SliderFloat(
                "Field of view", &camera.fov, 15.0f, 100.0f, "%.1f deg", CLAMP);

            ImGui::SliderFloat(
                "Orthographic height", &camera.orthoHeight,
                1.0f, 40.0f, "%.2f", CLAMP);

            ImGui::SliderFloat(
                "Near plane", &camera.nearPlane,
                0.01f, 2.0f, "%.3f", CLAMP);

            ImGui::SliderFloat(
                "Far plane", &camera.farPlane,
                10.0f, 500.0f, "%.1f", CLAMP);

            ImGui::SliderFloat(
                "Movement speed", &camera.speed,
                0.2f, 20.0f, "%.2f", CLAMP);

            if (ImGui::Button("Reset camera")) {
                camera = Camera{};
            }

            ImGui::TextWrapped(
                "Outside UI: hold RMB to look and use W/A/S/D, Q/E. "
                "Shift accelerates. Mouse wheel changes FOV.");
        }

        void drawModelsUI() {
            const char* names[] = {
                "Dodecahedron 1",
                "Dodecahedron 2",
                "Floor"
            };

            for (uint32_t i = 0; i < MODEL_COUNT; ++i) {
                ImGui::PushID(static_cast<int>(i));

                if (ImGui::TreeNode(names[i])) {
                    auto& model = models[i];

                    ImGui::Checkbox("Visible", &model.visible);

                    ImGui::DragFloat3(
                        "Position", glm::value_ptr(model.position), 0.03f);

                    ImGui::SliderFloat3(
                        "Rotation", glm::value_ptr(model.rotation),
                        -180.0f, 180.0f, "%.1f", CLAMP);

                    ImGui::SliderFloat3(
                        "Scale", glm::value_ptr(model.scale),
                        0.05f, 15.0f, "%.2f", CLAMP);

                    ImGui::ColorEdit3(
                        "Albedo", glm::value_ptr(model.albedo));

                    ImGui::ColorEdit3(
                        "Specular color", glm::value_ptr(model.specular));

                    ImGui::SliderFloat(
                        "Shininess", &model.shininess,
                        1.0f, 256.0f, "%.1f", CLAMP);

                if (ImGui::BeginCombo("Texture material", materials[model.material].name.c_str())) {
                    for (int m=0; m<int(materials.size()); ++m) {
                        if (ImGui::Selectable(materials[m].name.c_str(), model.material==m)) model.material=m;
                    }
                    ImGui::EndCombo();
                }
                ImGui::SliderFloat("UV tiling", &model.uvScale, 0.1f, 12.0f, "%.2f", CLAMP);
                ImGui::Checkbox("Animated double sampling", &model.effect);
                ImGui::SliderFloat("Effect speed", &model.effectSpeed, 0.0f, 3.0f, "%.2f", CLAMP);
                ImGui::SliderFloat("Effect strength", &model.effectStrength, 0.0f, 1.0f, "%.2f", CLAMP);
                ImGui::SliderFloat("Emission strength", &model.emission, 0.0f, 5.0f, "%.2f", CLAMP);

                    ImGui::TreePop();
                }

                ImGui::PopID();
            }

            ImGui::Separator();

            ImGui::Checkbox("Animate dodecahedra", &animateModels);

            ImGui::SliderFloat(
                "Animation speed", &animationSpeed,
                0.0f, 4.0f, "%.2f", CLAMP);

            ImGui::SliderFloat(
                "Orbit radius", &orbitRadius, 0.0f, 3.0f, "%.2f", CLAMP);

            ImGui::SliderFloat(
                "Orbit height", &orbitHeight, 0.0f, 2.0f, "%.2f", CLAMP);

            ImGui::SliderFloat(
                "Orbit frequency", &orbitFrequency,
                0.1f, 3.0f, "%.2f", CLAMP);

            ImGui::SliderFloat(
                "Rotation speed", &rotationSpeed,
                0.0f, 180.0f, "%.1f deg/s", CLAMP);

            if (ImGui::Button("Reset animation")) {
                animationTime = 0.0;
            }
        }

        void drawPointsUI() {
            ImGui::Text("Count: %u", static_cast<unsigned>(points.size()));

            ImGui::BeginDisabled(points.size() >= maxStorageRange / sizeof(PointGpu));
            if (ImGui::Button("Add point light")) {
                points.emplace_back();
            }
            ImGui::EndDisabled();

            int removeIndex = -1;

            for (size_t i = 0; i < points.size(); ++i) {
                ImGui::PushID(static_cast<int>(i));

                const std::string label = "Point " + std::to_string(i + 1);

                if (ImGui::TreeNode(label.c_str())) {
                    auto& light = points[i];

                    ImGui::DragFloat3(
                        "Position", glm::value_ptr(light.position), 0.05f);

                    ImGui::ColorEdit3(
                        "Color", glm::value_ptr(light.colorIntensity));

                    ImGui::SliderFloat(
                        "Intensity", &light.colorIntensity.w,
                        0.0f, 150.0f, "%.2f", CLAMP);

                    if (ImGui::Button("Delete point light")) {
                        removeIndex = static_cast<int>(i);
                    }

                    ImGui::TreePop();
                }

                ImGui::PopID();
            }

            if (removeIndex >= 0) {
                points.erase(points.begin() + removeIndex);
            }
        }

        void drawSpotsUI() {
            ImGui::Text("Count: %u", static_cast<unsigned>(spots.size()));

            ImGui::BeginDisabled(spots.size() >= maxStorageRange / sizeof(SpotGpu));
            if (ImGui::Button("Add spotlight")) {
                spots.emplace_back();
            }
            ImGui::EndDisabled();

            int removeIndex = -1;

            for (size_t i = 0; i < spots.size(); ++i) {
                ImGui::PushID(static_cast<int>(i));

                const std::string label = "Spot " + std::to_string(i + 1);

                if (ImGui::TreeNode(label.c_str())) {
                    auto& light = spots[i];

                    ImGui::DragFloat3(
                        "Position", glm::value_ptr(light.gpu.position), 0.05f);

                    ImGui::DragFloat3(
                        "Direction",
                        glm::value_ptr(light.gpu.directionCosOuter), 0.02f);

                    if (ImGui::Button("Aim at first object")) {
                        const vec3 direction =
                            models[0].position - vec3(light.gpu.position);

                        light.gpu.directionCosOuter = vec4(
                            safeNormalize(direction), 0.0f);
                    }

                    ImGui::ColorEdit3(
                        "Color", glm::value_ptr(light.gpu.colorIntensity));

                    ImGui::SliderFloat(
                        "Intensity", &light.gpu.colorIntensity.w,
                        0.0f, 200.0f, "%.2f", CLAMP);

                    ImGui::SliderFloat(
                        "Inner half-angle", &light.innerAngle,
                        0.0f, 85.0f, "%.1f deg", CLAMP);

                    ImGui::SliderFloat(
                        "Outer half-angle", &light.outerAngle,
                        0.0f, 85.0f, "%.1f deg", CLAMP);

                    light.outerAngle = std::max(
                        light.outerAngle, light.innerAngle);

                    ImGui::TextWrapped(
                        "Angles are measured from the cone axis. "
                        "Equal angles produce a hard edge.");

                    if (ImGui::Button("Delete spotlight")) {
                        removeIndex = static_cast<int>(i);
                    }

                    ImGui::TreePop();
                }

                ImGui::PopID();
            }

            if (removeIndex >= 0) {
                spots.erase(spots.begin() + removeIndex);
            }
        }

        mat4 modelMatrix(uint32_t index) {
            const auto& model = models[index];

            vec3 position = model.position;
            vec3 rotation = model.rotation;

            if (index < 2) {
                const float phase = static_cast<float>(
                    std::remainder(
                        animationTime * static_cast<double>(orbitFrequency),
                        2.0 * static_cast<double>(PI)));

                const float sign = index == 0 ? 1.0f : -0.65f;

                // At t = 0 the orbit offset is zero.
                position += sign * vec3(
                    (std::cos(phase) - 1.0f) * orbitRadius,
                    std::sin(2.0f * phase) * orbitHeight,
                    std::sin(phase) * orbitRadius);

                const double turn = animationTime * rotationSpeed;

                rotation += vec3(
                    static_cast<float>(std::remainder(turn * sign * 0.7, 360.0)),
                    static_cast<float>(std::remainder(turn * sign, 360.0)),
                    static_cast<float>(std::remainder(turn * sign * 0.35, 360.0)));
            }

            mat4 result = glm::translate(mat4(1.0f), position);

            result = glm::rotate(
                result, glm::radians(rotation.z), vec3(0, 0, 1));
            result = glm::rotate(
                result, glm::radians(rotation.y), vec3(0, 1, 0));
            result = glm::rotate(
                result, glm::radians(rotation.x), vec3(1, 0, 0));

            return glm::scale(result, model.scale);
        }

        void drawMesh(VkCommandBuffer command, const Mesh& mesh) {
            const VkDeviceSize offset = 0;

            vkCmdBindVertexBuffers(
                command, 0, 1, &mesh.vertices.handle, &offset);

            vkCmdBindIndexBuffer(command, mesh.indices.handle, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(command, mesh.count, 1, 0, 0, 0);
        }

        void setPush(VkCommandBuffer command, const Push& push) {
            vkCmdPushConstants(
                command,
                pipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(Push),
                &push);
        }

        void drawMarker(
            VkCommandBuffer command,
            vec3 position,
            vec3 color) {

            ModelGpu marker{};
            marker.options.x = 1;
            marker.albedo = vec4(color, 1.0f);

            marker.model = glm::scale(
                glm::translate(mat4(1.0f), position),
                vec3(markerSize));

            writeModel(drawSlot, marker);
            bindMaterial(command, 0, drawSlot++);
            setPush(command, Push{});
            drawMesh(command, cubeMesh);
        }

        void drawSpotCone(
            VkCommandBuffer command,
            const Spot& light,
            float angle,
            vec3 color) {

            const vec3 direction = safeNormalize(
                vec3(light.gpu.directionCosOuter));

            const vec3 reference =
                std::abs(direction.y) < 0.95f
                ? vec3(0, 1, 0) : vec3(1, 0, 0);

            const vec3 right = safeNormalize(
                glm::cross(direction, reference), vec3(1, 0, 0));

            const vec3 up = safeNormalize(
                glm::cross(right, direction), vec3(0, 1, 0));

            mat4 orientation(1.0f);
            orientation[0] = vec4(right, 0.0f);
            orientation[1] = vec4(up, 0.0f);
            orientation[2] = vec4(-direction, 0.0f);

            const float radius =
                coneLength * std::tan(glm::radians(angle));

            ModelGpu marker{};
            marker.options.x = 1;
            marker.albedo = vec4(color, 1.0f);

            marker.model =
                glm::translate(mat4(1.0f), vec3(light.gpu.position)) *
                orientation *
                glm::scale(mat4(1.0f), vec3(radius, radius, coneLength));

            writeModel(drawSlot, marker);
            bindMaterial(command, 0, drawSlot++);
            setPush(command, Push{});
            drawMesh(command, coneMesh);
        }

    } // namespace

    bool initialize() {
        try {
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(
                graphics::internal::context.physical_device,
                &properties);

            maxStorageRange = properties.limits.maxStorageBufferRange;

            if (properties.limits.maxPushConstantsSize < sizeof(Push)) {
                throw std::runtime_error("Insufficient push constant size");
            }

            models[0] = Model{};

            models[1] = Model{};
            models[1].position = vec3(-3.3f, -0.3f, -0.5f);
            models[1].scale = vec3(0.8f);
            models[1].albedo = vec3(1.0f, 0.35f, 0.16f);
            models[1].shininess = 32.0f;

            models[2] = Model{};
            models[2].position = vec3(0.0f, -1.9f, 0.0f);
            models[2].scale = vec3(14.0f, 1.0f, 14.0f);
            models[2].albedo = vec3(0.32f, 0.36f, 0.43f);
            models[2].specular = vec3(0.25f);
            models[2].shininess = 24.0f;

            points = {
                {
                    vec4(-3.0f, 2.5f, 2.0f, 1.0f),
                    vec4(1.0f, 0.25f, 0.12f, 28.0f)
                },
                {
                    vec4(3.0f, 1.5f, 1.0f, 1.0f),
                    vec4(0.15f, 0.5f, 1.0f, 25.0f)
                }
            };

            spots = { Spot{} };

            dodecahedron = createMesh(createDodecahedron());
            floorMesh = createMesh(createFloor());
            cubeMesh = createMesh(createCube());
            coneMesh = createMesh(createConeLines());

            sceneBuffer = createBuffer(
                sizeof(SceneGpu), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

            pointBuffer = createBuffer(
                sizeof(PointGpu), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

            spotBuffer = createBuffer(
                sizeof(SpotGpu), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

            const VkDeviceSize alignment = properties.limits.minUniformBufferOffsetAlignment;
            modelStride = (sizeof(ModelGpu) + alignment - 1) / alignment * alignment;
            modelBuffer = createBuffer(modelStride * 32, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
            loadMaterials();
            models[0].albedo = models[1].albedo = models[2].albedo = vec3(1);
            models[0].material = 1;
            models[1].material = 2;
            models[1].effect = true;
            models[2].uvScale = 5.0f;
            createDescriptors();
            createPipelines();

            return true;
        }
        catch (const std::exception& error) {
            std::cerr << "Application initialization: "
                << error.what() << '\n';
            shutdown();
            return false;
        }
    }

    void shutdown() {
        const auto device = graphics::internal::context.device;

        if (!device) {
            return;
        }

        vkDeviceWaitIdle(device);

        if (linePipeline) {
            vkDestroyPipeline(device, linePipeline, nullptr);
            linePipeline = VK_NULL_HANDLE;
        }

        if (trianglePipeline) {
            vkDestroyPipeline(device, trianglePipeline, nullptr);
            trianglePipeline = VK_NULL_HANDLE;
        }

        if (pipelineLayout) {
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            pipelineLayout = VK_NULL_HANDLE;
        }

        if (descriptorPool) {
            vkDestroyDescriptorPool(device, descriptorPool, nullptr);
            descriptorPool = VK_NULL_HANDLE;
        }

        if (modelLayout) {
            vkDestroyDescriptorSetLayout(device, modelLayout, nullptr);
            modelLayout = VK_NULL_HANDLE;
        }

        if (sceneLayout) {
            vkDestroyDescriptorSetLayout(device, sceneLayout, nullptr);
            sceneLayout = VK_NULL_HANDLE;
        }

        destroyBuffer(modelBuffer);
        destroyTextures();

        destroyBuffer(sceneBuffer);
        destroyBuffer(pointBuffer);
        destroyBuffer(spotBuffer);

        destroyBuffer(dodecahedron.vertices);
        destroyBuffer(dodecahedron.indices);
        destroyBuffer(floorMesh.vertices);
        destroyBuffer(floorMesh.indices);
        destroyBuffer(cubeMesh.vertices);
        destroyBuffer(cubeMesh.indices);
        destroyBuffer(coneMesh.vertices);
        destroyBuffer(coneMesh.indices);
    }

    void update(double time) {
        const float deltaTime = lastTime < 0.0
            ? 0.0f
            : static_cast<float>(std::clamp(time - lastTime, 0.0, 0.05));

        lastTime = time;

        updateCameraInput(deltaTime);
        if (animateTextures) textureTime += deltaTime;

        ImGui::SetNextWindowPos(ImVec2(12, 12), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(405, 760), ImGuiCond_FirstUseEver);

        ImGui::Begin("Lab 3 - Texture Mapping");

        ImGui::Text("Lab 3: Texture mapping / Variant 10");
        ImGui::Checkbox("Animate texture effects", &animateTextures);
        if (ImGui::Button("All lights OFF (emissive test)")) {
            sunEnabled=false; ambientIntensity=0;
            for (auto& p: points) p.colorIntensity.w=0;
            for (auto& p: spots) p.gpu.colorIntensity.w=0;
        }
        if (ImGui::Button("Restore lighting")) {
            sunEnabled=true; sunIntensity=0.7f; ambientIntensity=0.08f;
            for (auto& p: points) p.colorIntensity.w=25;
            for (auto& p: spots) p.gpu.colorIntensity.w=60;
        }
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);

        if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
            drawCameraUI();
        }

        if (ImGui::CollapsingHeader("Objects and materials")) {
            drawModelsUI();
        }

        if (ImGui::CollapsingHeader("Ambient and directional light")) {
            ImGui::ColorEdit3(
                "Ambient color", glm::value_ptr(ambientColor));

            ImGui::SliderFloat(
                "Ambient intensity", &ambientIntensity,
                0.0f, 1.0f, "%.3f", CLAMP);

            ImGui::Separator();

            ImGui::Checkbox("Enable directional light", &sunEnabled);

            ImGui::DragFloat3(
                "Light direction", glm::value_ptr(sunDirection), 0.02f);

            ImGui::ColorEdit3("Light color", glm::value_ptr(sunColor));

            ImGui::SliderFloat(
                "Directional intensity", &sunIntensity,
                0.0f, 5.0f, "%.2f", CLAMP);

            ImGui::TextWrapped(
                "Direction points FROM the light. "
                "A zero direction falls back to (0, -1, 0).");
        }

        if (ImGui::CollapsingHeader("Point lights")) {
            drawPointsUI();
        }

        if (ImGui::CollapsingHeader("Spotlights")) {
            drawSpotsUI();
        }

        if (ImGui::CollapsingHeader("Light visualization")) {
            ImGui::Checkbox("Show light cubes", &showMarkers);
            ImGui::Checkbox("Show spotlight cones", &showCones);

            ImGui::SliderFloat(
                "Cube size", &markerSize,
                0.03f, 0.6f, "%.2f", CLAMP);

            ImGui::SliderFloat(
                "Cone display length", &coneLength,
                0.2f, 8.0f, "%.2f", CLAMP);

            ImGui::TextWrapped(
                "White: outer cone. Gray: inner cone. "
                "Cone length is only a visualization parameter.");
        }

        ImGui::End();

        if (animateModels) {
            animationTime += double(deltaTime) * animationSpeed;
        }
    }

    void render(const graphics::internal::FrameData& frame) {
        auto& context = graphics::internal::context;
        const VkCommandBuffer command = frame.command_buffer;
        ensureModelCapacity(MODEL_COUNT + points.size() + 3 * spots.size());
        drawSlot = MODEL_COUNT;

        vec3 forward, right, up;
        cameraBasis(forward, right, up);

        const mat4 view = glm::lookAtRH(
            camera.position, camera.position + forward, up);

        const float aspect =
            float(context.swapchain_extent.width) /
            float(context.swapchain_extent.height);

        mat4 projection;

        if (camera.perspective) {
            projection = glm::perspectiveRH_ZO(
                glm::radians(camera.fov),
                aspect,
                camera.nearPlane,
                camera.farPlane);
        }
        else {
            const float halfHeight = camera.orthoHeight * 0.5f;
            const float halfWidth = halfHeight * aspect;

            projection = glm::orthoRH_ZO(
                -halfWidth, halfWidth,
                -halfHeight, halfHeight,
                camera.nearPlane, camera.farPlane);
        }

        // Vulkan viewport Y direction.
        projection[1][1] *= -1.0f;

        std::vector<SpotGpu> spotData;
        spotData.reserve(spots.size());

        for (const auto& spot : spots) {
            SpotGpu gpu = spot.gpu;

            gpu.directionCosOuter = vec4(
                safeNormalize(vec3(spot.gpu.directionCosOuter)),
                std::cos(glm::radians(spot.outerAngle)));

            gpu.cone.x = std::cos(glm::radians(spot.innerAngle));
            spotData.push_back(gpu);
        }

        uploadLights(pointBuffer, 1, points);
        uploadLights(spotBuffer, 2, spotData);

        SceneGpu scene{};
        scene.viewProjection = projection * view;
        scene.cameraPosition = vec4(camera.position, 1.0f);
        scene.ambient = vec4(ambientColor, ambientIntensity);
        scene.sunDirection = vec4(safeNormalize(sunDirection), 0.0f);
        scene.sunColorIntensity = vec4(
            sunColor, sunEnabled ? sunIntensity : 0.0f);

        scene.counts = glm::uvec4(
            static_cast<uint32_t>(points.size()),
            static_cast<uint32_t>(spots.size()),
            0, 0);

        upload(sceneBuffer, &scene, sizeof(scene));

        for (uint32_t i = 0; i < MODEL_COUNT; ++i) {
            ModelGpu gpu{};
            gpu.model = modelMatrix(i);

            // Correct even for non-uniform object scaling.
            gpu.normalMatrix = glm::transpose(glm::inverse(gpu.model));

            gpu.albedo = vec4(models[i].albedo, 1.0f);
            gpu.specularShininess = vec4(
                models[i].specular, models[i].shininess);

            gpu.options = vec4(0.0f, models[i].emission, 0.0f, 0.0f);

            writeModel(i, gpu);
        }

        VkViewport viewport{};
        viewport.width = float(context.swapchain_extent.width);
        viewport.height = float(context.swapchain_extent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.extent = context.swapchain_extent;

        vkCmdSetViewport(command, 0, 1, &viewport);
        vkCmdSetScissor(command, 0, 1, &scissor);

        vkCmdBindPipeline(
            command, VK_PIPELINE_BIND_POINT_GRAPHICS, trianglePipeline);

        vkCmdBindDescriptorSets(
            command,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipelineLayout,
            0, 1, &sceneSet,
            0, nullptr);

        Push litPush{};
        setPush(command, litPush);

        for (uint32_t i = 0; i < MODEL_COUNT; ++i) {
            if (!models[i].visible) {
                continue;
            }

            bindMaterial(command, models[i].material, i);
            Push effect{};
            effect.flags.x = models[i].effect ? 1 : 0;
            effect.params = vec4(textureTime * models[i].effectSpeed, models[i].uvScale,
                                 models[i].effectStrength, 0.0f);
            setPush(command, effect);

            drawMesh(command, i == 2 ? floorMesh : dodecahedron);
        }

        if (showMarkers) {
            for (const auto& light : points) {
                drawMarker(
                    command,
                    vec3(light.position),
                    vec3(light.colorIntensity));
            }

            for (const auto& light : spots) {
                drawMarker(
                    command,
                    vec3(light.gpu.position),
                    vec3(light.gpu.colorIntensity));
            }
        }

        if (showCones) {
            vkCmdBindPipeline(
                command, VK_PIPELINE_BIND_POINT_GRAPHICS, linePipeline);

            for (const auto& light : spots) {
                drawSpotCone(
                    command, light, light.outerAngle, vec3(1.0f));

                if (light.innerAngle + 0.01f < light.outerAngle) {
                    drawSpotCone(
                        command, light, light.innerAngle, vec3(0.45f));
                }
            }
        }

        // The graphics module draws ImGui and ends the render pass.
    }

} // namespace application
