#version 450

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;

layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;
layout(location = 2) out vec3 proceduralColor;

layout(set = 0, binding = 0, std140) uniform SceneBlock
{
    mat4 viewProjection;
    vec4 cameraPosition;
    vec4 ambient;
    vec4 sunDirection;
    vec4 sunColorIntensity;
    uvec4 counts;
} scene;

layout(set = 1, binding = 0, std140) uniform ModelBlock
{
    mat4 model;
    mat4 normalMatrix;
    vec4 albedo;
    vec4 specularShininess;
    vec4 options;
} object;

layout(push_constant) uniform PushBlock
{
    mat4 transform;
    vec4 color;
    ivec4 flags;
} pushData;

vec3 safeNormalize(vec3 value, vec3 fallback)
{
    float lengthSquared = dot(value, value);

    return lengthSquared > 1e-12
        ? value * inversesqrt(lengthSquared)
        : fallback;
}

void main()
{
    bool unlit = pushData.flags.x != 0;

    mat4 transform = unlit
        ? pushData.transform
        : object.model;

    vec4 world = transform * vec4(aPosition, 1.0);

    worldPosition = world.xyz;

    worldNormal = unlit
        ? vec3(0.0, 1.0, 0.0)
        : mat3(object.normalMatrix) * aNormal;

    // The original lab's position-dependent vertex colors.
    proceduralColor =
        safeNormalize(aPosition, vec3(0.0, 1.0, 0.0)) * 0.5 + 0.5;

    gl_Position = scene.viewProjection * world;
}