#include "application.hpp"

#include <imgui.h>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace application {

	namespace {

		// ============================================================
		// Math
		// ============================================================

		constexpr float PI = 3.14159265358979323846f;

		struct Vec3 {
			float x;
			float y;
			float z;
		};

		struct Mat4 {
			float m[16]{};

			float& at(int row, int column) {
				return m[column * 4 + row];
			}

			const float& at(int row, int column) const {
				return m[column * 4 + row];
			}
		};

		Mat4 identity() {
			Mat4 result{};

			result.at(0, 0) = 1.0f;
			result.at(1, 1) = 1.0f;
			result.at(2, 2) = 1.0f;
			result.at(3, 3) = 1.0f;

			return result;
		}

		Mat4 multiply(const Mat4& a, const Mat4& b) {
			Mat4 result{};

			for (int row = 0; row < 4; ++row) {
				for (int column = 0; column < 4; ++column) {
					float value = 0.0f;

					for (int k = 0; k < 4; ++k) {
						value += a.at(row, k) * b.at(k, column);
					}

					result.at(row, column) = value;
				}
			}

			return result;
		}

		Mat4 translation(float x, float y, float z) {
			Mat4 result = identity();

			result.at(0, 3) = x;
			result.at(1, 3) = y;
			result.at(2, 3) = z;

			return result;
		}

		Mat4 scale(float x, float y, float z) {
			Mat4 result{};

			result.at(0, 0) = x;
			result.at(1, 1) = y;
			result.at(2, 2) = z;
			result.at(3, 3) = 1.0f;

			return result;
		}

		Mat4 rotationX(float angle) {
			const float c = std::cos(angle);
			const float s = std::sin(angle);

			Mat4 result = identity();

			result.at(1, 1) = c;
			result.at(1, 2) = -s;
			result.at(2, 1) = s;
			result.at(2, 2) = c;

			return result;
		}

		Mat4 rotationY(float angle) {
			const float c = std::cos(angle);
			const float s = std::sin(angle);

			Mat4 result = identity();

			result.at(0, 0) = c;
			result.at(0, 2) = s;
			result.at(2, 0) = -s;
			result.at(2, 2) = c;

			return result;
		}

		Mat4 rotationZ(float angle) {
			const float c = std::cos(angle);
			const float s = std::sin(angle);

			Mat4 result = identity();

			result.at(0, 0) = c;
			result.at(0, 1) = -s;
			result.at(1, 0) = s;
			result.at(1, 1) = c;

			return result;
		}

		// Vulkan perspective projection.
		// Camera looks along -Z.
		// Z range is [0, 1], as required by Vulkan.
		Mat4 perspective(float fovRadians, float aspect, float nearPlane, float farPlane) {
			const float f = 1.0f / std::tan(fovRadians * 0.5f);

			Mat4 result{};

			result.at(0, 0) = f / aspect;

			// Vulkan Y flip.
			result.at(1, 1) = -f;

			result.at(2, 2) = farPlane / (nearPlane - farPlane);
			result.at(2, 3) = (farPlane * nearPlane) / (nearPlane - farPlane);
			result.at(3, 2) = -1.0f;

			return result;
		}

		Mat4 orthographic(
			float left,
			float right,
			float bottom,
			float top,
			float nearPlane,
			float farPlane) {

			Mat4 result = identity();

			result.at(0, 0) = 2.0f / (right - left);
			result.at(1, 1) = -2.0f / (top - bottom);

			result.at(2, 2) = 1.0f / (nearPlane - farPlane);

			result.at(0, 3) = -(right + left) / (right - left);
			result.at(1, 3) = -(top + bottom) / (top - bottom);
			result.at(2, 3) = nearPlane / (nearPlane - farPlane);

			return result;
		}

		float radians(float degrees) {
			return degrees * PI / 180.0f;
		}

		// ============================================================
		// Dodecahedron
		// ============================================================

		struct Vertex {
			float position[3];
		};

		// Regular dodecahedron:
		//
		// (±1, ±1, ±1)
		// (0, ±1/phi, ±phi)
		// (±1/phi, ±phi, 0)
		// (±phi, 0, ±1/phi)
		//
		// phi = (1 + sqrt(5)) / 2
		//
		constexpr float PHI = 1.6180339887498948482f;
		constexpr float INV_PHI = 0.6180339887498948482f;

		const std::array<Vertex, 20> g_vertices = { {
			{{-1.0f, -1.0f, -1.0f}},
			{{-1.0f, -1.0f,  1.0f}},
			{{-1.0f,  1.0f, -1.0f}},
			{{-1.0f,  1.0f,  1.0f}},
			{{ 1.0f, -1.0f, -1.0f}},
			{{ 1.0f, -1.0f,  1.0f}},
			{{ 1.0f,  1.0f, -1.0f}},
			{{ 1.0f,  1.0f,  1.0f}},

			{{ 0.0f, -INV_PHI, -PHI}},
			{{ 0.0f, -INV_PHI,  PHI}},
			{{ 0.0f,  INV_PHI, -PHI}},
			{{ 0.0f,  INV_PHI,  PHI}},

			{{-INV_PHI, -PHI, 0.0f}},
			{{-INV_PHI,  PHI, 0.0f}},
			{{ INV_PHI, -PHI, 0.0f}},
			{{ INV_PHI,  PHI, 0.0f}},

			{{-PHI, 0.0f, -INV_PHI}},
			{{ PHI, 0.0f, -INV_PHI}},
			{{-PHI, 0.0f,  INV_PHI}},
			{{ PHI, 0.0f,  INV_PHI}}
		} };

		// 12 pentagonal faces.
		// Every pentagon is triangulated as:
		// (0,1,2), (0,2,3), (0,3,4).
		//
		// The order is chosen so that the normals point outward.
		const std::array<std::array<uint32_t, 5>, 12> g_faces = { {
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

		std::array<uint32_t, 108> createIndices() {
			std::array<uint32_t, 108> indices{};

			size_t index = 0;

			for (const auto& face : g_faces) {
				indices[index++] = face[0];
				indices[index++] = face[1];
				indices[index++] = face[2];

				indices[index++] = face[0];
				indices[index++] = face[2];
				indices[index++] = face[3];

				indices[index++] = face[0];
				indices[index++] = face[3];
				indices[index++] = face[4];
			}

			return indices;
		}

		// ============================================================
		// GPU resources
		// ============================================================

		struct Buffer {
			VkBuffer buffer = VK_NULL_HANDLE;
			VmaAllocation allocation = VK_NULL_HANDLE;
			void* mapped = nullptr;
			VkDeviceSize size = 0;
		};

		constexpr uint32_t OBJECT_COUNT = 2;

		struct UniformBufferObject {
			Mat4 mvp;
			float color[4];
		};

		Buffer g_vertexBuffer;
		Buffer g_indexBuffer;
		std::array<Buffer, OBJECT_COUNT> g_uniformBuffers;

		VkDescriptorSetLayout g_descriptorSetLayout = VK_NULL_HANDLE;
		VkDescriptorPool g_descriptorPool = VK_NULL_HANDLE;
		std::array<VkDescriptorSet, OBJECT_COUNT> g_descriptorSets{};

		VkPipelineLayout g_pipelineLayout = VK_NULL_HANDLE;
		VkPipeline g_pipeline = VK_NULL_HANDLE;

		uint32_t g_indexCount = 108;

		// ============================================================
		// ImGUI state
		// ============================================================

		bool g_perspective = true;

		float g_position[3] = {
			0.0f,
			0.0f,
			0.0f
		};

		float g_rotation[3] = {
			0.0f,
			0.0f,
			0.0f
		};

		float g_scale[3] = {
			1.0f,
			1.0f,
			1.0f
		};

		float g_color[4] = {
			0.2f,
			0.65f,
			1.0f,
			1.0f
		};

		bool g_animationEnabled = true;
		bool g_showSecondObject = true;

		float g_animationSpeed = 1.0f;
		float g_orbitRadius = 1.5f;
		float g_orbitHeight = 0.8f;
		float g_orbitFrequency = 1.0f;
		float g_rotationSpeed = 45.0f;

		double g_lastTime = 0.0;
		float g_animationTime = 0.0f;

		// ============================================================
		// Vulkan helpers
		// ============================================================

		void destroyBuffer(Buffer& buffer) {
			auto& context = graphics::internal::context;

			if (buffer.buffer != VK_NULL_HANDLE) {
				vmaDestroyBuffer(
					context.allocator,
					buffer.buffer,
					buffer.allocation);
			}

			buffer.buffer = VK_NULL_HANDLE;
			buffer.allocation = VK_NULL_HANDLE;
			buffer.mapped = nullptr;
			buffer.size = 0;
		}

		bool createBuffer(
			VkDeviceSize size,
			VkBufferUsageFlags usage,
			Buffer& result) {

			auto& context = graphics::internal::context;

			const VkBufferCreateInfo bufferInfo = {
				.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
				.size = size,
				.usage = usage,
				.sharingMode = VK_SHARING_MODE_EXCLUSIVE
			};

			const VmaAllocationCreateInfo allocationInfo = {
				.flags =
					VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
					VMA_ALLOCATION_CREATE_MAPPED_BIT,
				.usage = VMA_MEMORY_USAGE_AUTO
			};

			VmaAllocationInfo actualAllocationInfo{};

			const VkResult vkResult = vmaCreateBuffer(
				context.allocator,
				&bufferInfo,
				&allocationInfo,
				&result.buffer,
				&result.allocation,
				&actualAllocationInfo);

			if (vkResult != VK_SUCCESS) {
				std::cerr << "Failed to create VMA buffer. VkResult = "
					<< vkResult << '\n';

				return false;
			}

			result.size = size;
			result.mapped = actualAllocationInfo.pMappedData;

			return true;
		}

		bool loadShaderModule(
			const char* filename,
			VkShaderModule& shaderModule) {

			std::ifstream file(
				filename,
				std::ios::ate | std::ios::binary);

			if (!file.is_open()) {
				std::cerr << "Failed to open shader: "
					<< filename << '\n';

				return false;
			}

			const std::streamsize fileSize = file.tellg();

			if (fileSize <= 0 || fileSize % 4 != 0) {
				std::cerr << "Invalid SPIR-V file: "
					<< filename << '\n';

				return false;
			}

			file.seekg(0);

			std::vector<char> bytes(
				static_cast<size_t>(fileSize));

			file.read(bytes.data(), fileSize);

			if (!file) {
				std::cerr << "Failed to read shader: "
					<< filename << '\n';

				return false;
			}

			std::vector<uint32_t> code(
				static_cast<size_t>(fileSize) / sizeof(uint32_t));

			std::memcpy(
				code.data(),
				bytes.data(),
				bytes.size());

			auto& context = graphics::internal::context;

			const VkShaderModuleCreateInfo createInfo = {
				.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
				.codeSize = code.size() * sizeof(uint32_t),
				.pCode = code.data()
			};

			if (vkCreateShaderModule(
				context.device,
				&createInfo,
				nullptr,
				&shaderModule) != VK_SUCCESS) {

				std::cerr << "Failed to create shader module: "
					<< filename << '\n';

				return false;
			}

			return true;
		}

		// ============================================================
		// Pipeline
		// ============================================================

		bool createDescriptorResources() {
			auto& context = graphics::internal::context;

			const VkDescriptorSetLayoutBinding uboBinding = {
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_VERTEX_BIT
			};

			const VkDescriptorSetLayoutCreateInfo layoutInfo = {
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				.bindingCount = 1,
				.pBindings = &uboBinding
			};

			if (vkCreateDescriptorSetLayout(
				context.device,
				&layoutInfo,
				nullptr,
				&g_descriptorSetLayout) != VK_SUCCESS) {

				std::cerr << "Failed to create descriptor set layout\n";
				return false;
			}

			const VkDescriptorPoolSize poolSize = {
				.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = OBJECT_COUNT
			};

			const VkDescriptorPoolCreateInfo poolInfo = {
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
				.maxSets = OBJECT_COUNT,
				.poolSizeCount = 1,
				.pPoolSizes = &poolSize
			};

			if (vkCreateDescriptorPool(
				context.device,
				&poolInfo,
				nullptr,
				&g_descriptorPool) != VK_SUCCESS) {

				std::cerr << "Failed to create descriptor pool\n";
				return false;
			}

			std::array<VkDescriptorSetLayout, OBJECT_COUNT> layouts{};

			for (auto& layout : layouts) {
				layout = g_descriptorSetLayout;
			}

			const VkDescriptorSetAllocateInfo allocateInfo = {
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = g_descriptorPool,
				.descriptorSetCount = OBJECT_COUNT,
				.pSetLayouts = layouts.data()
			};

			if (vkAllocateDescriptorSets(
				context.device,
				&allocateInfo,
				g_descriptorSets.data()) != VK_SUCCESS) {

				std::cerr << "Failed to allocate descriptor sets\n";
				return false;
			}

			for (uint32_t i = 0; i < OBJECT_COUNT; ++i) {
				const VkDescriptorBufferInfo bufferInfo = {
					.buffer = g_uniformBuffers[i].buffer,
					.offset = 0,
					.range = sizeof(UniformBufferObject)
				};

				const VkWriteDescriptorSet write = {
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = g_descriptorSets[i],
					.dstBinding = 0,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.pBufferInfo = &bufferInfo
				};

				vkUpdateDescriptorSets(
					context.device,
					1,
					&write,
					0,
					nullptr);
			}

			return true;
		}

		bool createPipeline() {
			auto& context = graphics::internal::context;

			VkShaderModule vertexShader = VK_NULL_HANDLE;
			VkShaderModule fragmentShader = VK_NULL_HANDLE;

#ifdef SHADER_DIR
			const char* vertexShaderPath =
				SHADER_DIR "/basic.vert.spv";

			const char* fragmentShaderPath =
				SHADER_DIR "/basic.frag.spv";
#else
			const char* vertexShaderPath =
				"shaders/basic.vert.spv";

			const char* fragmentShaderPath =
				"shaders/basic.frag.spv";
#endif

			if (!loadShaderModule(vertexShaderPath, vertexShader)) {
				return false;
			}

			if (!loadShaderModule(fragmentShaderPath, fragmentShader)) {
				vkDestroyShaderModule(
					context.device,
					vertexShader,
					nullptr);

				return false;
			}

			const VkPipelineShaderStageCreateInfo shaderStages[] = {
				{
					.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
					.stage = VK_SHADER_STAGE_VERTEX_BIT,
					.module = vertexShader,
					.pName = "main"
				},
				{
					.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
					.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
					.module = fragmentShader,
					.pName = "main"
				}
			};

			const VkVertexInputBindingDescription vertexBinding = {
				.binding = 0,
				.stride = sizeof(Vertex),
				.inputRate = VK_VERTEX_INPUT_RATE_VERTEX
			};

			const VkVertexInputAttributeDescription vertexAttribute = {
				.location = 0,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, position)
			};

			const VkPipelineVertexInputStateCreateInfo vertexInput = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
				.vertexBindingDescriptionCount = 1,
				.pVertexBindingDescriptions = &vertexBinding,
				.vertexAttributeDescriptionCount = 1,
				.pVertexAttributeDescriptions = &vertexAttribute
			};

			const VkPipelineInputAssemblyStateCreateInfo inputAssembly = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
				.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
				.primitiveRestartEnable = VK_FALSE
			};

			const VkPipelineViewportStateCreateInfo viewportState = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
				.viewportCount = 1,
				.scissorCount = 1
			};

			const VkPipelineRasterizationStateCreateInfo rasterization = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
				.depthClampEnable = VK_FALSE,
				.rasterizerDiscardEnable = VK_FALSE,
				.polygonMode = VK_POLYGON_MODE_FILL,
				.cullMode = VK_CULL_MODE_BACK_BIT,

				// The projection matrix contains the Vulkan Y flip.
				// Therefore the winding is inverted in framebuffer space.
				.frontFace = VK_FRONT_FACE_CLOCKWISE,

				.depthBiasEnable = VK_FALSE,
				.lineWidth = 1.0f
			};

			const VkPipelineMultisampleStateCreateInfo multisample = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
				.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT
			};

			const VkPipelineDepthStencilStateCreateInfo depthStencil = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
				.depthTestEnable = VK_TRUE,
				.depthWriteEnable = VK_TRUE,
				.depthCompareOp = VK_COMPARE_OP_LESS,
				.depthBoundsTestEnable = VK_FALSE,
				.stencilTestEnable = VK_FALSE
			};

			const VkPipelineColorBlendAttachmentState colorBlendAttachment = {
				.blendEnable = VK_FALSE,
				.colorWriteMask =
					VK_COLOR_COMPONENT_R_BIT |
					VK_COLOR_COMPONENT_G_BIT |
					VK_COLOR_COMPONENT_B_BIT |
					VK_COLOR_COMPONENT_A_BIT
			};

			const VkPipelineColorBlendStateCreateInfo colorBlend = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
				.logicOpEnable = VK_FALSE,
				.attachmentCount = 1,
				.pAttachments = &colorBlendAttachment
			};

			const VkDynamicState dynamicStates[] = {
				VK_DYNAMIC_STATE_VIEWPORT,
				VK_DYNAMIC_STATE_SCISSOR
			};

			const VkPipelineDynamicStateCreateInfo dynamicState = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
				.dynamicStateCount = 2,
				.pDynamicStates = dynamicStates
			};

			const VkPushConstantRange* pushConstants = nullptr;

			const VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
				.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
				.setLayoutCount = 1,
				.pSetLayouts = &g_descriptorSetLayout,
				.pushConstantRangeCount = 0,
				.pPushConstantRanges = pushConstants
			};

			if (vkCreatePipelineLayout(
				context.device,
				&pipelineLayoutInfo,
				nullptr,
				&g_pipelineLayout) != VK_SUCCESS) {

				std::cerr << "Failed to create pipeline layout\n";

				vkDestroyShaderModule(
					context.device,
					fragmentShader,
					nullptr);

				vkDestroyShaderModule(
					context.device,
					vertexShader,
					nullptr);

				return false;
			}

			const VkGraphicsPipelineCreateInfo pipelineInfo = {
				.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
				.stageCount = 2,
				.pStages = shaderStages,
				.pVertexInputState = &vertexInput,
				.pInputAssemblyState = &inputAssembly,
				.pViewportState = &viewportState,
				.pRasterizationState = &rasterization,
				.pMultisampleState = &multisample,
				.pDepthStencilState = &depthStencil,
				.pColorBlendState = &colorBlend,
				.pDynamicState = &dynamicState,
				.layout = g_pipelineLayout,
				.renderPass = context.render_pass,
				.subpass = 0
			};

			if (vkCreateGraphicsPipelines(
				context.device,
				VK_NULL_HANDLE,
				1,
				&pipelineInfo,
				nullptr,
				&g_pipeline) != VK_SUCCESS) {

				std::cerr << "Failed to create graphics pipeline\n";

				vkDestroyPipelineLayout(
					context.device,
					g_pipelineLayout,
					nullptr);

				g_pipelineLayout = VK_NULL_HANDLE;

				vkDestroyShaderModule(
					context.device,
					fragmentShader,
					nullptr);

				vkDestroyShaderModule(
					context.device,
					vertexShader,
					nullptr);

				return false;
			}

			vkDestroyShaderModule(
				context.device,
				fragmentShader,
				nullptr);

			vkDestroyShaderModule(
				context.device,
				vertexShader,
				nullptr);

			return true;
		}

		// ============================================================
		// Cleanup
		// ============================================================

		void destroyResources() {
			auto& context = graphics::internal::context;

			if (context.device == VK_NULL_HANDLE) {
				return;
			}

			vkDeviceWaitIdle(context.device);

			if (g_pipeline != VK_NULL_HANDLE) {
				vkDestroyPipeline(
					context.device,
					g_pipeline,
					nullptr);

				g_pipeline = VK_NULL_HANDLE;
			}

			if (g_pipelineLayout != VK_NULL_HANDLE) {
				vkDestroyPipelineLayout(
					context.device,
					g_pipelineLayout,
					nullptr);

				g_pipelineLayout = VK_NULL_HANDLE;
			}

			if (g_descriptorPool != VK_NULL_HANDLE) {
				vkDestroyDescriptorPool(
					context.device,
					g_descriptorPool,
					nullptr);

				g_descriptorPool = VK_NULL_HANDLE;
			}

			if (g_descriptorSetLayout != VK_NULL_HANDLE) {
				vkDestroyDescriptorSetLayout(
					context.device,
					g_descriptorSetLayout,
					nullptr);

				g_descriptorSetLayout = VK_NULL_HANDLE;
			}

			for (auto& buffer : g_uniformBuffers) {
				destroyBuffer(buffer);
			}

			destroyBuffer(g_indexBuffer);
			destroyBuffer(g_vertexBuffer);

			for (auto& descriptorSet : g_descriptorSets) {
				descriptorSet = VK_NULL_HANDLE;
			}
		}

		// ============================================================
		// Per-frame data
		// ============================================================

		Mat4 createModelMatrix(
			float x,
			float y,
			float z,
			float rotationXDegrees,
			float rotationYDegrees,
			float rotationZDegrees,
			float scaleX,
			float scaleY,
			float scaleZ) {

			const Mat4 translationMatrix =
				translation(x, y, z);

			const Mat4 scaleMatrix =
				scale(scaleX, scaleY, scaleZ);

			const Mat4 rx =
				rotationX(radians(rotationXDegrees));

			const Mat4 ry =
				rotationY(radians(rotationYDegrees));

			const Mat4 rz =
				rotationZ(radians(rotationZDegrees));

			const Mat4 rotation =
				multiply(
					multiply(rz, ry),
					rx);

			return multiply(
				translationMatrix,
				multiply(rotation, scaleMatrix));
		}

		void updateUniformBuffer(
			uint32_t objectIndex,
			const Mat4& mvp) {

			UniformBufferObject ubo{};

			ubo.mvp = mvp;

			ubo.color[0] = g_color[0];
			ubo.color[1] = g_color[1];
			ubo.color[2] = g_color[2];
			ubo.color[3] = g_color[3];

			std::memcpy(
				g_uniformBuffers[objectIndex].mapped,
				&ubo,
				sizeof(ubo));

			vmaFlushAllocation(
				graphics::internal::context.allocator,
				g_uniformBuffers[objectIndex].allocation,
				0,
				sizeof(ubo));
		}

	} // namespace

	// ============================================================
	// Public application API
	// ============================================================

	bool initialize() {
		auto& context = graphics::internal::context;

		const auto indices = createIndices();

		// --------------------------------------------------------
		// Vertex buffer
		// --------------------------------------------------------

		if (!createBuffer(
			sizeof(g_vertices),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			g_vertexBuffer)) {

			destroyResources();
			return false;
		}

		std::memcpy(
			g_vertexBuffer.mapped,
			g_vertices.data(),
			sizeof(g_vertices));

		vmaFlushAllocation(
			context.allocator,
			g_vertexBuffer.allocation,
			0,
			sizeof(g_vertices));

		// --------------------------------------------------------
		// Index buffer
		// --------------------------------------------------------

		if (!createBuffer(
			sizeof(indices),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			g_indexBuffer)) {

			destroyResources();
			return false;
		}

		std::memcpy(
			g_indexBuffer.mapped,
			indices.data(),
			sizeof(indices));

		vmaFlushAllocation(
			context.allocator,
			g_indexBuffer.allocation,
			0,
			sizeof(indices));

		// --------------------------------------------------------
		// Two uniform buffers = two descriptor sets
		// --------------------------------------------------------

		for (uint32_t i = 0; i < OBJECT_COUNT; ++i) {
			if (!createBuffer(
				sizeof(UniformBufferObject),
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				g_uniformBuffers[i])) {

				destroyResources();
				return false;
			}
		}

		if (!createDescriptorResources()) {
			destroyResources();
			return false;
		}

		if (!createPipeline()) {
			destroyResources();
			return false;
		}

		g_indexCount = static_cast<uint32_t>(indices.size());

		g_lastTime = 0.0;
		g_animationTime = 0.0f;

		return true;
	}

	void shutdown() {
		destroyResources();
	}

	void update(double time) {
		// --------------------------------------------------------
		// Animation time
		// --------------------------------------------------------

		const double deltaTime =
			std::max(0.0, time - g_lastTime);

		g_lastTime = time;

		if (g_animationEnabled) {
			g_animationTime +=
				static_cast<float>(deltaTime) *
				g_animationSpeed;
		}

		// --------------------------------------------------------
		// ImGUI
		// --------------------------------------------------------

		ImGui::Begin("Dodecahedron - Lab #1");

		ImGui::Text("Variant 10: Regular Dodecahedron");
		ImGui::Text("20 vertices / 12 faces / 30 edges");

		ImGui::Separator();

		ImGui::Text("Projection");

		if (ImGui::RadioButton(
			"Perspective",
			g_perspective)) {

			g_perspective = true;
		}

		ImGui::SameLine();

		if (ImGui::RadioButton(
			"Orthographic",
			!g_perspective)) {

			g_perspective = false;
		}

		ImGui::Separator();

		ImGui::Text("Transform");

		ImGui::SliderFloat3(
			"Position",
			g_position,
			-5.0f,
			5.0f);

		ImGui::SliderFloat3(
			"Rotation",
			g_rotation,
			-180.0f,
			180.0f,
			"%.1f deg");

		ImGui::SliderFloat3(
			"Scale",
			g_scale,
			0.1f,
			3.0f);

		ImGui::Separator();

		ImGui::Text("Animation");

		ImGui::Checkbox(
			"Play animation",
			&g_animationEnabled);

		ImGui::SliderFloat(
			"Speed",
			&g_animationSpeed,
			0.0f,
			5.0f);

		ImGui::SliderFloat(
			"Orbit radius",
			&g_orbitRadius,
			0.0f,
			4.0f);

		ImGui::SliderFloat(
			"Orbit height",
			&g_orbitHeight,
			0.0f,
			3.0f);

		ImGui::SliderFloat(
			"Orbit frequency",
			&g_orbitFrequency,
			0.1f,
			3.0f);

		ImGui::SliderFloat(
			"Rotation speed",
			&g_rotationSpeed,
			0.0f,
			180.0f);

		if (ImGui::Button("Reset animation")) {
			g_animationTime = 0.0f;
		}

		ImGui::Separator();

		ImGui::Text("Appearance");

		ImGui::ColorEdit4(
			"Object color",
			g_color);

		ImGui::Checkbox(
			"Show second object",
			&g_showSecondObject);

		ImGui::End();
	}

	void render(const graphics::internal::FrameData& fd) {
		auto& context = graphics::internal::context;

		// --------------------------------------------------------
		// Command buffer
		// --------------------------------------------------------

		const VkCommandBufferBeginInfo commandBufferBegin = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
		};

		if (vkBeginCommandBuffer(
			fd.command_buffer,
			&commandBufferBegin) != VK_SUCCESS) {

			std::cerr << "vkBeginCommandBuffer failed\n";
			return;
		}

		// --------------------------------------------------------
		// Render pass
		// --------------------------------------------------------

		const VkClearValue clearValues[] = {
			{
				.color = {
					.float32 = {
						0.015f,
						0.02f,
						0.035f,
						1.0f
					}
				}
			},
			{
				.depthStencil = {
					.depth = 1.0f,
					.stencil = 0
				}
			}
		};

		const VkRenderPassBeginInfo renderPassBegin = {
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = context.render_pass,
			.framebuffer = fd.framebuffer,
			.renderArea = {
				.offset = { 0, 0 },
				.extent = context.swapchain_extent
			},
			.clearValueCount = 2,
			.pClearValues = clearValues
		};

		vkCmdBeginRenderPass(
			fd.command_buffer,
			&renderPassBegin,
			VK_SUBPASS_CONTENTS_INLINE);

		// --------------------------------------------------------
		// Dynamic viewport/scissor
		// --------------------------------------------------------

		const VkViewport viewport = {
			.x = 0.0f,
			.y = 0.0f,
			.width =
				static_cast<float>(context.swapchain_extent.width),
			.height =
				static_cast<float>(context.swapchain_extent.height),
			.minDepth = 0.0f,
			.maxDepth = 1.0f
		};

		const VkRect2D scissor = {
			.offset = { 0, 0 },
			.extent = context.swapchain_extent
		};

		vkCmdSetViewport(
			fd.command_buffer,
			0,
			1,
			&viewport);

		vkCmdSetScissor(
			fd.command_buffer,
			0,
			1,
			&scissor);

		// --------------------------------------------------------
		// Pipeline
		// --------------------------------------------------------

		vkCmdBindPipeline(
			fd.command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			g_pipeline);

		const VkDeviceSize vertexOffset = 0;

		vkCmdBindVertexBuffers(
			fd.command_buffer,
			0,
			1,
			&g_vertexBuffer.buffer,
			&vertexOffset);

		vkCmdBindIndexBuffer(
			fd.command_buffer,
			g_indexBuffer.buffer,
			0,
			VK_INDEX_TYPE_UINT32);

		// --------------------------------------------------------
		// Camera / projection
		// --------------------------------------------------------

		const float width =
			static_cast<float>(context.swapchain_extent.width);

		const float height =
			static_cast<float>(context.swapchain_extent.height);

		const float aspect =
			height > 0.0f
			? width / height
			: 1.0f;

		Mat4 projection;

		if (g_perspective) {
			projection = perspective(
				radians(55.0f),
				aspect,
				0.1f,
				100.0f);
		}
		else {
			const float orthoHeight = 7.0f;
			const float orthoWidth =
				orthoHeight * aspect;

			projection = orthographic(
				-orthoWidth,
				orthoWidth,
				-orthoHeight,
				orthoHeight,
				0.1f,
				100.0f);
		}

		// Camera is at (0, 0, +8) looking towards -Z.
		const Mat4 view =
			translation(0.0f, 0.0f, -8.0f);

		// --------------------------------------------------------
		// Object 1 animation
		// --------------------------------------------------------

		const float t =
			g_animationTime * g_orbitFrequency;

		const float orbitX =
			std::cos(t) * g_orbitRadius;

		const float orbitZ =
			std::sin(t) * g_orbitRadius;

		const float orbitY =
			std::sin(t * 2.0f) * g_orbitHeight;

		const float rotationAnimation =
			g_animationTime * g_rotationSpeed;

		const Mat4 model1 = createModelMatrix(
			g_position[0] + orbitX,
			g_position[1] + orbitY,
			g_position[2] + orbitZ,

			g_rotation[0] + rotationAnimation * 0.7f,
			g_rotation[1] + rotationAnimation,
			g_rotation[2] + rotationAnimation * 0.35f,

			g_scale[0],
			g_scale[1],
			g_scale[2]);

		const Mat4 mvp1 =
			multiply(
				projection,
				multiply(view, model1));

		updateUniformBuffer(
			0,
			mvp1);

		// --------------------------------------------------------
		// Draw object 1
		// --------------------------------------------------------

		vkCmdBindDescriptorSets(
			fd.command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			g_pipelineLayout,
			0,
			1,
			&g_descriptorSets[0],
			0,
			nullptr);

		vkCmdDrawIndexed(
			fd.command_buffer,
			g_indexCount,
			1,
			0,
			0,
			0);

		// --------------------------------------------------------
		// Object 2
		// --------------------------------------------------------

		if (g_showSecondObject) {
			const Mat4 model2 = createModelMatrix(
				g_position[0] - 3.2f - orbitX * 0.65f,
				g_position[1] - orbitY * 0.5f,
				g_position[2] - orbitZ * 0.65f,

				g_rotation[0] - rotationAnimation * 0.5f,
				g_rotation[1] - rotationAnimation * 0.8f,
				g_rotation[2] + rotationAnimation * 0.25f,

				g_scale[0] * 0.85f,
				g_scale[1] * 0.85f,
				g_scale[2] * 0.85f);

			const Mat4 mvp2 =
				multiply(
					projection,
					multiply(view, model2));

			updateUniformBuffer(
				1,
				mvp2);

			vkCmdBindDescriptorSets(
				fd.command_buffer,
				VK_PIPELINE_BIND_POINT_GRAPHICS,
				g_pipelineLayout,
				0,
				1,
				&g_descriptorSets[1],
				0,
				nullptr);

			vkCmdDrawIndexed(
				fd.command_buffer,
				g_indexCount,
				1,
				0,
				0,
				0);
		}

		// --------------------------------------------------------
		// Finish render pass / command buffer
		// --------------------------------------------------------

		vkCmdEndRenderPass(
			fd.command_buffer);

		if (vkEndCommandBuffer(
			fd.command_buffer) != VK_SUCCESS) {

			std::cerr << "vkEndCommandBuffer failed\n";
			return;
		}
	}

} // namespace application