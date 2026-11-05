#version 450

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec2 texCoord;

layout(location = 0) out vec4 outColor;

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

struct PointLight
{
    vec4 position;
    vec4 colorIntensity;
};

struct SpotLight
{
    vec4 position;

    // xyz: normalized direction FROM the source.
    // w: cosine of the outer half-angle.
    vec4 directionCosOuter;

    vec4 colorIntensity;

    // x: cosine of the inner half-angle.
    vec4 cone;
};

// Runtime-sized arrays: no hardcoded maximum number of lights.
layout(set = 0, binding = 1, std430) readonly buffer PointBuffer
{
    PointLight lights[];
} pointBuffer;

layout(set = 0, binding = 2, std430) readonly buffer SpotBuffer
{
    SpotLight lights[];
} spotBuffer;

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

layout(set=1, binding=1) uniform sampler2D albedoMap;
layout(set=1, binding=2) uniform sampler2D specularMap;
layout(set=1, binding=3) uniform sampler2D emissiveMap;
vec3 sampledSpecular;

vec3 blinnPhong(
    vec3 normal,
    vec3 viewDirection,
    vec3 lightDirection,
    vec3 radiance,
    vec3 albedo)
{
    float nDotL = max(dot(normal, lightDirection), 0.0);

    // A source behind the surface contributes no diffuse or specular.
    if (nDotL <= 0.0)
    {
        return vec3(0.0);
    }

    vec3 halfVector = safeNormalize(
        lightDirection + viewDirection,
        vec3(0.0));

    float specularFactor = pow(
        max(dot(normal, halfVector), 0.0),
        max(object.specularShininess.w, 1.0));

    vec3 diffuse = albedo * nDotL;

    vec3 specular =
        object.specularShininess.rgb * sampledSpecular * specularFactor;

    return (diffuse + specular) * radiance;
}

void main()
{
    // Light visualization bypasses the illumination model entirely.
    if (object.options.x > 0.5)
    {
        outColor = vec4(object.albedo.rgb, 1.0);
        return;
    }

    vec3 normal = safeNormalize(
        worldNormal, vec3(0.0, 1.0, 0.0));

    vec3 viewDirection = safeNormalize(
        scene.cameraPosition.xyz - worldPosition,
        normal);

    vec2 uv = texCoord * pushData.params.y;
    vec3 albedoSample = texture(albedoMap, uv).rgb;
    sampledSpecular = texture(specularMap, uv).rgb;
    vec3 emission = texture(emissiveMap, uv).rgb;
    if (pushData.flags.x != 0) {
        float t = pushData.params.x;
        float strength = pushData.params.z;
        vec2 uvA = uv + vec2(0.08, 0.03) * t;
        vec2 uvB = uv * 1.6 + vec2(-0.04, 0.07) * t;
        uvB += 0.04 * sin(uv.yx * 6.2831853 + t);
        vec3 a = texture(albedoMap, uvA).rgb;
        vec3 b = texture(albedoMap, uvB).rgb;
        vec3 modulation = 0.65 + 0.35 * sin(vec3(0.0, 2.1, 4.2) + t);
        albedoSample = mix(albedoSample, mix(a,b,0.5)*modulation, strength);
    }
    vec3 albedo = object.albedo.rgb * albedoSample;

    vec3 result =
        albedo * scene.ambient.rgb * scene.ambient.w;

    // Directional source: direction is stored FROM the source,
    // while the illumination model needs the direction TO the source.
    vec3 sunL = safeNormalize(
        -scene.sunDirection.xyz, vec3(0.0, 1.0, 0.0));

    result += blinnPhong(
        normal,
        viewDirection,
        sunL,
        scene.sunColorIntensity.rgb * scene.sunColorIntensity.w,
        albedo);

    uint pointCount = min(
        scene.counts.x,
        uint(pointBuffer.lights.length()));

    for (uint i = 0u; i < pointCount; ++i)
    {
        PointLight light = pointBuffer.lights[i];

        vec3 toLight = light.position.xyz - worldPosition;
        float distanceSquared = dot(toLight, toLight);

        vec3 lightDirection = safeNormalize(
            toLight, normal);

        // Inverse-square attenuation.
        // The small lower bound avoids the singularity at distance = 0.
        float attenuation = 1.0 / max(distanceSquared, 0.0001);

        vec3 radiance =
            light.colorIntensity.rgb *
            light.colorIntensity.w *
            attenuation;

        result += blinnPhong(
            normal,
            viewDirection,
            lightDirection,
            radiance,
            albedo);
    }

    uint spotCount = min(
        scene.counts.y,
        uint(spotBuffer.lights.length()));

    for (uint i = 0u; i < spotCount; ++i)
    {
        SpotLight light = spotBuffer.lights[i];

        vec3 toLight = light.position.xyz - worldPosition;
        float distanceSquared = dot(toLight, toLight);

        vec3 lightDirection = safeNormalize(
            toLight, normal);

        vec3 spotDirection = safeNormalize(
            light.directionCosOuter.xyz,
            vec3(0.0, -1.0, 0.0));

        float theta = dot(-lightDirection, spotDirection);

        float cosOuter = light.directionCosOuter.w;
        float cosInner = light.cone.x;

        float coneFactor;

        if (cosInner > cosOuter + 1e-6)
        {
            coneFactor = smoothstep(cosOuter, cosInner, theta);
        }
        else
        {
            // Avoid division by zero when both angles are equal.
            coneFactor = step(cosOuter, theta);
        }

        float attenuation = 1.0 / max(distanceSquared, 0.0001);

        vec3 radiance =
            light.colorIntensity.rgb *
            light.colorIntensity.w *
            attenuation *
            coneFactor;

        result += blinnPhong(
            normal,
            viewDirection,
            lightDirection,
            radiance,
            albedo);
    }

    // Lighting is computed in linear space.
    // The sRGB swapchain performs the output conversion.
    outColor = vec4(max(result + emission * object.options.y, vec3(0.0)), 1.0);
}