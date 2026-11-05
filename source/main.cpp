#include <cstdint>
#include <climits>
#include <cstdlib>
#include <exception>
#include <string_view>
#include <backends/imgui_impl_vulkan.h>

#include <iostream>


#include <GLFW/glfw3.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>

#include "graphics_internal.hpp"
#include "application.hpp"

namespace {

constexpr int32_t default_window_width = 1280;
constexpr int32_t default_window_height = 720;

constexpr char default_window_title[] = "CG Lab 3 - Texture Mapping";

GLFWwindow* glfw_window;

} // namespace

int main(int argc, char** argv) {
	int status = EXIT_SUCCESS;

	if (!glfwInit()) {
		std::cerr << "Failed to initialize GLFW\n";
		return EXIT_FAILURE;
	}

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    if (argc > 1 && std::string_view(argv[1]) == "--smoke-test") glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

	glfw_window = glfwCreateWindow(default_window_width, default_window_height,
	                               default_window_title, nullptr, nullptr);
	if (glfw_window == nullptr) {
		status = EXIT_FAILURE;
		goto err_null_window;
	}

	glfwSetFramebufferSizeCallback(glfw_window, [](GLFWwindow*, int width, int height){
		if (width == 0 || height == 0) {
			return;
		}

		graphics::internal::resize(width, height);
	});

	if (ImGui::CreateContext() == nullptr) {
		std::cerr << "Failed to create ImGUI context\n";
		status = EXIT_FAILURE;
		goto err_imgui_init;
	}

	if (!ImGui_ImplGlfw_InitForVulkan(glfw_window, true)) {
		std::cerr << "Failed to initialize ImGUI GLFW backend for Vulkan renderer\n";
		status = EXIT_FAILURE;
		goto err_imgui_glfw_init;
	}

	if (!graphics::internal::initialize(glfw_window)) {
		std::cerr << "Failed to initialize graphics\n";
		status = EXIT_FAILURE;
		goto err_graphics_init;
	}

	if (!application::initialize()) {
		std::cerr << "Failed to initialize application\n";
		status = EXIT_FAILURE;
		goto err_application_init;
	}

    try {
        const bool smoke = argc > 1 && std::string_view(argv[1]) == "--smoke-test";
        int frames = 0;
        while (!glfwWindowShouldClose(glfw_window)) {
            glfwPollEvents();
            int width=0, height=0;
            glfwGetFramebufferSize(glfw_window, &width, &height);
            if (width<=0 || height<=0) { glfwWaitEventsTimeout(0.1); continue; }
            auto fd=graphics::internal::prepare();
            if (!fd.command_buffer) continue;
            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            application::update(glfwGetTime());
            ImGui::Render();
            application::render(fd);
            graphics::internal::submitAndPresent();
            if (smoke && ++frames == 30) glfwSetWindowSize(glfw_window, 960, 600);
            if (smoke && frames >= 90) glfwSetWindowShouldClose(glfw_window, GLFW_TRUE);
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        status=EXIT_FAILURE;
    }

	application::shutdown();
err_application_init:
	graphics::internal::shutdown();
err_graphics_init:
	ImGui_ImplGlfw_Shutdown();
err_imgui_glfw_init:
	ImGui::DestroyContext();
err_imgui_init:
	glfwDestroyWindow(glfw_window);
err_null_window:
	glfwTerminate();

	return status;
}

