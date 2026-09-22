#version 450

layout(location = 0) in vec3 vColor;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform ObjectUBO
{
    mat4 mvp;
    vec4 color;
} ubo;

void main()
{
    // ImGUI-selected color is multiplied by procedural vertex color.
    vec3 finalColor =
        vColor * ubo.color.rgb;

    outColor = vec4(
        finalColor,
        ubo.color.a);
}