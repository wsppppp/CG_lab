#version 450

layout(location = 0) in vec3 aPosition;

layout(location = 0) out vec3 vColor;

layout(set = 0, binding = 0) uniform ObjectUBO
{
    mat4 mvp;
    vec4 color;
} ubo;

void main()
{
    gl_Position = ubo.mvp * vec4(aPosition, 1.0);

    // Procedural vertex color based on local-space position.
    vec3 normalizedPosition = normalize(aPosition);

    vColor = normalizedPosition * 0.5 + 0.5;
}