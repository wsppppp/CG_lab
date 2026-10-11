#define VMA_IMPLEMENTATION
#include "graphics_internal.hpp"

#include <GLFW/glfw3.h>
#include <VkBootstrap.h>

#include <imgui.h>
#include <backends/imgui_impl_vulkan.h>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace graphics::internal {

    Context context{};

    namespace {

        GLFWwindow* windowHandle = nullptr;

        VkInstance instance = VK_NULL_HANDLE;
        VkSurfaceKHR surface = VK_NULL_HANDLE;

        VkQueue presentQueue = VK_NULL_HANDLE;
        uint32_t presentQueueFamily = 0;

        VkSwapchainKHR swapchain = VK_NULL_HANDLE;
        std::vector<VkImageView> imageViews;
        std::vector<VkFramebuffer> framebuffers;
        std::vector<VkSemaphore> renderFinished;

        VkImage depthImage = VK_NULL_HANDLE;
        VmaAllocation depthAllocation = VK_NULL_HANDLE;
        VkImageView depthView = VK_NULL_HANDLE;
        VkFormat depthFormat = VK_FORMAT_UNDEFINED;

        VkCommandPool commandPool = VK_NULL_HANDLE;
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        VkFence frameFence = VK_NULL_HANDLE;

        uint32_t imageIndex = 0;

        bool swapchainDirty = false;
        bool imguiInitialized = false;

        void check(VkResult result, const char* operation) {
            if (result != VK_SUCCESS) {
                throw std::runtime_error(
                    std::string(operation) + ": VkResult = " +
                    std::to_string(static_cast<int>(result)));
            }
        }

        VkFormat selectDepthFormat() {
            for (VkFormat format : {
                VK_FORMAT_D32_SFLOAT,
                    VK_FORMAT_D24_UNORM_S8_UINT,
                    VK_FORMAT_D16_UNORM
            }) {
                VkFormatProperties properties{};
                vkGetPhysicalDeviceFormatProperties(
                    context.physical_device, format, &properties);

                if (properties.optimalTilingFeatures &
                    VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
                    return format;
                }
            }

            throw std::runtime_error("No supported depth format");
        }

        void createRenderPass() {
            VkAttachmentDescription attachments[2]{};

            attachments[0].format = context.swapchain_format;
            attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
            attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

            attachments[1].format = depthFormat;
            attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
            attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            attachments[1].finalLayout =
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            VkAttachmentReference colorReference{};
            colorReference.attachment = 0;
            colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            VkAttachmentReference depthReference{};
            depthReference.attachment = 1;
            depthReference.layout =
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            VkSubpassDescription subpass{};
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments = &colorReference;
            subpass.pDepthStencilAttachment = &depthReference;

            VkSubpassDependency dependency{};
            dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
            dependency.dstSubpass = 0;

            dependency.srcStageMask =
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;

            dependency.dstStageMask = dependency.srcStageMask;

            dependency.srcAccessMask =
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

            dependency.dstAccessMask =
                VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

            dependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

            VkRenderPassCreateInfo info{
                VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO
            };

            info.attachmentCount = 2;
            info.pAttachments = attachments;
            info.subpassCount = 1;
            info.pSubpasses = &subpass;
            info.dependencyCount = 1;
            info.pDependencies = &dependency;

            check(vkCreateRenderPass(
                context.device, &info, nullptr, &context.render_pass),
                "vkCreateRenderPass");
        }

        void destroySwapchainResources() {
            for (VkFramebuffer framebuffer : framebuffers) {
                if (framebuffer) {
                    vkDestroyFramebuffer(context.device, framebuffer, nullptr);
                }
            }
            framebuffers.clear();

            for (VkSemaphore semaphore : renderFinished) {
                if (semaphore) {
                    vkDestroySemaphore(context.device, semaphore, nullptr);
                }
            }
            renderFinished.clear();

            if (depthView) {
                vkDestroyImageView(context.device, depthView, nullptr);
                depthView = VK_NULL_HANDLE;
            }

            if (depthImage) {
                vmaDestroyImage(context.allocator, depthImage, depthAllocation);
                depthImage = VK_NULL_HANDLE;
                depthAllocation = VK_NULL_HANDLE;
            }

            for (VkImageView view : imageViews) {
                if (view) {
                    vkDestroyImageView(context.device, view, nullptr);
                }
            }
            imageViews.clear();

            if (swapchain) {
                vkDestroySwapchainKHR(context.device, swapchain, nullptr);
                swapchain = VK_NULL_HANDLE;
            }
        }

        void initializeImGuiRenderer() {
            ImGui_ImplVulkan_InitInfo info{};

            info.ApiVersion = VK_API_VERSION_1_1;
            info.Instance = instance;
            info.PhysicalDevice = context.physical_device;
            info.Device = context.device;
            info.QueueFamily = context.graphics_queue_index;
            info.Queue = context.graphics_queue;

            // The backend creates and owns its descriptor pool.
            info.DescriptorPoolSize = 64;

            info.MinImageCount = 2;
            info.ImageCount = static_cast<uint32_t>(imageViews.size());

            info.PipelineInfoMain.RenderPass = context.render_pass;
            info.PipelineInfoMain.Subpass = 0;
            info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

            info.CheckVkResultFn = [](VkResult result) {
                check(result, "ImGui Vulkan backend");
                };

            const bool success = ImGui_ImplVulkan_Init(&info);
            imguiInitialized =
                ImGui::GetIO().BackendRendererUserData != nullptr;

            if (!success) {
                throw std::runtime_error("ImGui Vulkan initialization failed");
            }
        }

        bool rebuildSwapchain() {
            int width = 0;
            int height = 0;
            glfwGetFramebufferSize(windowHandle, &width, &height);

            if (width <= 0 || height <= 0) {
                return false;
            }

            check(vkDeviceWaitIdle(context.device), "vkDeviceWaitIdle");

            vkb::SwapchainBuilder builder(
                context.physical_device,
                context.device,
                surface,
                context.graphics_queue_index,
                presentQueueFamily);

            auto result = builder
                .set_old_swapchain(swapchain)
                .set_desired_extent(
                    static_cast<uint32_t>(width),
                    static_cast<uint32_t>(height))
                .set_desired_format({
                    VK_FORMAT_B8G8R8A8_SRGB,
                    VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
                    })
                .add_fallback_format({
                    VK_FORMAT_R8G8B8A8_SRGB,
                    VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
                    })
                .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
                .set_required_min_image_count(2)
                .set_desired_min_image_count(3)
                .set_image_usage_flags(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
                .build();

            if (!result) {
                throw std::runtime_error(result.error().message());
            }

            auto newSwapchain = result.value();

            const bool srgb =
                newSwapchain.image_format == VK_FORMAT_B8G8R8A8_SRGB ||
                newSwapchain.image_format == VK_FORMAT_R8G8B8A8_SRGB;

            if (!srgb) {
                vkb::destroy_swapchain(newSwapchain);
                throw std::runtime_error("An sRGB swapchain format is required");
            }

            if (context.render_pass &&
                context.swapchain_format != newSwapchain.image_format) {
                vkb::destroy_swapchain(newSwapchain);
                throw std::runtime_error(
                    "Surface format changed. Restart the application.");
            }

            if (imguiInitialized) {
                ImGui_ImplVulkan_Shutdown();
                imguiInitialized = false;
            }

            destroySwapchainResources();

            swapchain = newSwapchain.swapchain;
            context.swapchain_format = newSwapchain.image_format;
            context.swapchain_extent = newSwapchain.extent;

            auto viewsResult = newSwapchain.get_image_views();
            if (!viewsResult) {
                throw std::runtime_error(viewsResult.error().message());
            }
            imageViews = viewsResult.value();

            if (!context.render_pass) {
                createRenderPass();
            }

            VkImageCreateInfo imageInfo{
                VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO
            };

            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.format = depthFormat;
            imageInfo.extent = {
                context.swapchain_extent.width,
                context.swapchain_extent.height,
                1
            };
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            VmaAllocationCreateInfo allocationInfo{};
            allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;

            check(vmaCreateImage(
                context.allocator,
                &imageInfo,
                &allocationInfo,
                &depthImage,
                &depthAllocation,
                nullptr),
                "vmaCreateImage");

            VkImageViewCreateInfo viewInfo{
                VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO
            };

            viewInfo.image = depthImage;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = depthFormat;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.layerCount = 1;

            check(vkCreateImageView(
                context.device, &viewInfo, nullptr, &depthView),
                "vkCreateImageView");

            framebuffers.resize(imageViews.size(), VK_NULL_HANDLE);
            renderFinished.resize(imageViews.size(), VK_NULL_HANDLE);

            VkSemaphoreCreateInfo semaphoreInfo{
                VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
            };

            for (size_t i = 0; i < imageViews.size(); ++i) {
                const VkImageView attachments[] = {
                    imageViews[i], depthView
                };

                VkFramebufferCreateInfo framebufferInfo{
                    VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO
                };

                framebufferInfo.renderPass = context.render_pass;
                framebufferInfo.attachmentCount = 2;
                framebufferInfo.pAttachments = attachments;
                framebufferInfo.width = context.swapchain_extent.width;
                framebufferInfo.height = context.swapchain_extent.height;
                framebufferInfo.layers = 1;

                check(vkCreateFramebuffer(
                    context.device,
                    &framebufferInfo,
                    nullptr,
                    &framebuffers[i]),
                    "vkCreateFramebuffer");

                check(vkCreateSemaphore(
                    context.device,
                    &semaphoreInfo,
                    nullptr,
                    &renderFinished[i]),
                    "vkCreateSemaphore");
            }

            initializeImGuiRenderer();
            swapchainDirty = false;
            return true;
        }

    } // namespace

    bool initialize(GLFWwindow* window) {
        windowHandle = window;

        try {
            vkb::InstanceBuilder instanceBuilder;

            auto instanceResult = instanceBuilder
                .require_api_version(1, 1, 0)
                .request_validation_layers()
                .build();

            if (!instanceResult) {
                throw std::runtime_error(instanceResult.error().message());
            }

            const auto bootstrapInstance = instanceResult.value();
            instance = bootstrapInstance.instance;

            check(glfwCreateWindowSurface(
                instance, windowHandle, nullptr, &surface),
                "glfwCreateWindowSurface");

            vkb::PhysicalDeviceSelector selector(
                bootstrapInstance, surface);

            auto physicalResult = selector
                .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
                .require_present()
                .select();

            if (!physicalResult) {
                throw std::runtime_error(physicalResult.error().message());
            }

            vkb::DeviceBuilder deviceBuilder(physicalResult.value());
            auto deviceResult = deviceBuilder.build();

            if (!deviceResult) {
                throw std::runtime_error(deviceResult.error().message());
            }

            const auto bootstrapDevice = deviceResult.value();

            context.physical_device = bootstrapDevice.physical_device;
            context.device = bootstrapDevice.device;

            auto graphicsQueueResult =
                bootstrapDevice.get_queue(vkb::QueueType::graphics);
            auto graphicsFamilyResult =
                bootstrapDevice.get_queue_index(vkb::QueueType::graphics);
            auto presentQueueResult =
                bootstrapDevice.get_queue(vkb::QueueType::present);
            auto presentFamilyResult =
                bootstrapDevice.get_queue_index(vkb::QueueType::present);

            if (!graphicsQueueResult || !graphicsFamilyResult ||
                !presentQueueResult || !presentFamilyResult) {
                throw std::runtime_error("Required Vulkan queues unavailable");
            }

            context.graphics_queue = graphicsQueueResult.value();
            context.graphics_queue_index = graphicsFamilyResult.value();

            presentQueue = presentQueueResult.value();
            presentQueueFamily = presentFamilyResult.value();

            VmaAllocatorCreateInfo allocatorInfo{};
            allocatorInfo.instance = instance;
            allocatorInfo.physicalDevice = context.physical_device;
            allocatorInfo.device = context.device;
            allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_1;

            check(vmaCreateAllocator(
                &allocatorInfo, &context.allocator),
                "vmaCreateAllocator");

            depthFormat = selectDepthFormat();

            VkCommandPoolCreateInfo poolInfo{
                VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO
            };

            poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            poolInfo.queueFamilyIndex = context.graphics_queue_index;

            check(vkCreateCommandPool(
                context.device, &poolInfo, nullptr, &commandPool),
                "vkCreateCommandPool");

            VkCommandBufferAllocateInfo commandInfo{
                VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO
            };

            commandInfo.commandPool = commandPool;
            commandInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            commandInfo.commandBufferCount = 1;

            check(vkAllocateCommandBuffers(
                context.device, &commandInfo, &commandBuffer),
                "vkAllocateCommandBuffers");

            VkSemaphoreCreateInfo semaphoreInfo{
                VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
            };

            check(vkCreateSemaphore(
                context.device,
                &semaphoreInfo,
                nullptr,
                &imageAvailable),
                "vkCreateSemaphore");

            VkFenceCreateInfo fenceInfo{
                VK_STRUCTURE_TYPE_FENCE_CREATE_INFO
            };
            fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

            check(vkCreateFence(
                context.device, &fenceInfo, nullptr, &frameFence),
                "vkCreateFence");

            if (!rebuildSwapchain()) {
                throw std::runtime_error("Initial framebuffer size is zero");
            }

            return true;
        }
        catch (const std::exception& error) {
            std::cerr << "Graphics initialization: "
                << error.what() << '\n';
            shutdown();
            return false;
        }
    }

    void shutdown() {
        if (context.device) {
            vkDeviceWaitIdle(context.device);

            if (imguiInitialized) {
                ImGui_ImplVulkan_Shutdown();
                imguiInitialized = false;
            }

            if (commandPool) {
                vkDestroyCommandPool(context.device, commandPool, nullptr);
                commandPool = VK_NULL_HANDLE;
                commandBuffer = VK_NULL_HANDLE;
            }

            destroySwapchainResources();

            if (context.render_pass) {
                vkDestroyRenderPass(
                    context.device, context.render_pass, nullptr);
            }

            if (frameFence) {
                vkDestroyFence(context.device, frameFence, nullptr);
                frameFence = VK_NULL_HANDLE;
            }

            if (imageAvailable) {
                vkDestroySemaphore(context.device, imageAvailable, nullptr);
                imageAvailable = VK_NULL_HANDLE;
            }

            if (context.allocator) {
                vmaDestroyAllocator(context.allocator);
            }

            vkDestroyDevice(context.device, nullptr);
        }

        context = {};

        if (surface) {
            vkDestroySurfaceKHR(instance, surface, nullptr);
            surface = VK_NULL_HANDLE;
        }

        if (instance) {
            vkDestroyInstance(instance, nullptr);
            instance = VK_NULL_HANDLE;
        }

        windowHandle = nullptr;
        presentQueue = VK_NULL_HANDLE;
    }

    void resize(uint32_t, uint32_t) {
        // The actual framebuffer extent is queried during rebuilding.
        swapchainDirty = true;
    }

    FrameData prepare() {
        check(vkWaitForFences(
            context.device, 1, &frameFence, VK_TRUE, UINT64_MAX),
            "vkWaitForFences");

        if (swapchainDirty && !rebuildSwapchain()) {
            return {};
        }

        const VkResult result = vkAcquireNextImageKHR(
            context.device,
            swapchain,
            UINT64_MAX,
            imageAvailable,
            VK_NULL_HANDLE,
            &imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            swapchainDirty = true;
            return {};
        }

        if (result == VK_SUBOPTIMAL_KHR) {
            swapchainDirty = true;
        }
        else {
            check(result, "vkAcquireNextImageKHR");
        }

        check(vkResetCommandBuffer(commandBuffer, 0),
            "vkResetCommandBuffer");

        VkCommandBufferBeginInfo beginInfo{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        };
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        check(vkBeginCommandBuffer(commandBuffer, &beginInfo),
            "vkBeginCommandBuffer");

        VkClearValue clearValues[2]{};
        clearValues[0].color.float32[0] = 0.015f;
        clearValues[0].color.float32[1] = 0.020f;
        clearValues[0].color.float32[2] = 0.035f;
        clearValues[0].color.float32[3] = 1.0f;
        clearValues[1].depthStencil.depth = 1.0f;

        VkRenderPassBeginInfo renderInfo{
            VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO
        };

        renderInfo.renderPass = context.render_pass;
        renderInfo.framebuffer = framebuffers[imageIndex];
        renderInfo.renderArea.extent = context.swapchain_extent;
        renderInfo.clearValueCount = 2;
        renderInfo.pClearValues = clearValues;

        vkCmdBeginRenderPass(
            commandBuffer, &renderInfo, VK_SUBPASS_CONTENTS_INLINE);

        return { framebuffers[imageIndex], commandBuffer };
    }

    void submitAndPresent() {
        ImGui_ImplVulkan_RenderDrawData(
            ImGui::GetDrawData(), commandBuffer);

        vkCmdEndRenderPass(commandBuffer);
        check(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");

        const VkPipelineStageFlags waitStage =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

        VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };

        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &imageAvailable;
        submitInfo.pWaitDstStageMask = &waitStage;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &renderFinished[imageIndex];

        check(vkResetFences(context.device, 1, &frameFence),
            "vkResetFences");

        check(vkQueueSubmit(
            context.graphics_queue, 1, &submitInfo, frameFence),
            "vkQueueSubmit");

        VkPresentInfoKHR presentInfo{
            VK_STRUCTURE_TYPE_PRESENT_INFO_KHR
        };

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &renderFinished[imageIndex];
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain;
        presentInfo.pImageIndices = &imageIndex;

        const VkResult result = vkQueuePresentKHR(
            presentQueue, &presentInfo);

        if (result == VK_ERROR_OUT_OF_DATE_KHR ||
            result == VK_SUBOPTIMAL_KHR) {
            swapchainDirty = true;
        }
        else {
            check(result, "vkQueuePresentKHR");
        }
    }

} // namespace graphics::internal