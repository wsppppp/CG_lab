#version 450

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;

layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;
layout(location = 2) out vec2 texCoord;

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
    ivec4 flags;
    vec4 params;
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
    vec4 world = object.model * vec4(aPosition, 1.0);
    worldPosition = world.xyz;
    worldNormal = mat3(object.normalMatrix) * aNormal;
    texCoord = aUV;
    gl_Position = scene.viewProjection * world;
}