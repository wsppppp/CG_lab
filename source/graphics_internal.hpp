#pragma once

#include <cstdint>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

struct GLFWwindow;

namespace graphics::internal {

    struct Context {
        VkPhysicalDevice physical_device = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;

        VmaAllocator allocator = VK_NULL_HANDLE;

        VkQueue graphics_queue = VK_NULL_HANDLE;
        uint32_t graphics_queue_index = 0;

        VkFormat swapchain_format = VK_FORMAT_UNDEFINED;
        VkExtent2D swapchain_extent{};

        VkRenderPass render_pass = VK_NULL_HANDLE;
    };

    struct FrameData {
        VkFramebuffer framebuffer = VK_NULL_HANDLE;
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    };

    extern Context context;

    bool initialize(GLFWwindow* window);
    void shutdown();

    void resize(uint32_t width, uint32_t height);

    // Waits for the previous frame, acquires an image,
    // starts command recording and opens the render pass.
    // Returns an empty FrameData when this frame should be skipped.
    FrameData prepare();

    // Draws ImGui, closes the render pass and command buffer,
    // submits the frame and presents the image.
    void submitAndPresent();

} // namespace graphics::internal