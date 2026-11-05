cmake_minimum_required(VERSION 3.20)

project(vulkan-starter-app LANGUAGES C CXX)

include(FetchContent)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Vulkan REQUIRED)

set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
set(GLFW_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)
set(GLM_BUILD_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.5.1
    GIT_SHALLOW TRUE
)

FetchContent_Declare(
    vk-bootstrap
    GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git
    GIT_TAG v1.4.321
    GIT_SHALLOW TRUE
)

FetchContent_Declare(
    VulkanMemoryAllocator
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG v3.4.0
    GIT_SHALLOW TRUE
)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.92.9
    GIT_SHALLOW TRUE
)

FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(
    glfw
    vk-bootstrap
    VulkanMemoryAllocator
    imgui
    glm
)

add_library(imgui_lib STATIC
    "${imgui_SOURCE_DIR}/imgui.cpp"
    "${imgui_SOURCE_DIR}/imgui_draw.cpp"
    "${imgui_SOURCE_DIR}/imgui_tables.cpp"
    "${imgui_SOURCE_DIR}/imgui_widgets.cpp"
    "${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp"
    "${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp"
)

target_include_directories(imgui_lib PUBLIC
    "${imgui_SOURCE_DIR}"
    "${imgui_SOURCE_DIR}/backends"
)

target_link_libraries(imgui_lib PUBLIC
    glfw
    Vulkan::Vulkan
)

add_executable(${PROJECT_NAME}
    source/main.cpp
    source/application.cpp
    source/graphics_internal.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    imgui_lib
    glfw
    Vulkan::Vulkan
    vk-bootstrap::vk-bootstrap
    GPUOpen::VulkanMemoryAllocator
    glm::glm
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    NOMINMAX
    GLFW_INCLUDE_NONE
    GLM_FORCE_RADIANS
    GLM_FORCE_DEPTH_ZERO_TO_ONE
)

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4 /utf-8)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE
        -Wall
        -Wextra
        -Wno-missing-field-initializers
    )
endif()

find_program(
    GLSLC_EXECUTABLE
    NAMES glslc
    HINTS
        "$ENV{VULKAN_SDK}/Bin"
        "$ENV{VULKAN_SDK}/bin"
    REQUIRED
)

set(SHADER_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/shaders")

file(MAKE_DIRECTORY "${SHADER_OUTPUT_DIRECTORY}")

target_compile_definitions(${PROJECT_NAME} PRIVATE
    SHADER_DIR="${SHADER_OUTPUT_DIRECTORY}"
)

set(SHADER_BINARIES)

foreach(SHADER basic.vert basic.frag)
    set(INPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER}")
    set(OUTPUT_FILE "${SHADER_OUTPUT_DIRECTORY}/${SHADER}.spv")

    add_custom_command(
        OUTPUT "${OUTPUT_FILE}"
        COMMAND "${GLSLC_EXECUTABLE}"
            --target-env=vulkan1.1
            "${INPUT_FILE}"
            -o "${OUTPUT_FILE}"
        DEPENDS "${INPUT_FILE}"
        COMMENT "Compiling ${SHADER}"
        VERBATIM
    )

    list(APPEND SHADER_BINARIES "${OUTPUT_FILE}")
endforeach()

add_custom_target(shaders DEPENDS ${SHADER_BINARIES})
add_dependencies(${PROJECT_NAME} shaders)

target_compile_definitions(${PROJECT_NAME} PRIVATE TEXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/textures")
