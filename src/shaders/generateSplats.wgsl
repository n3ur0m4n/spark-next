// Native counterpart of SplatAccumulator's ordinary SplatMesh generation.
// One dispatch per mesh writes its upstream row-aligned destination range.
// Depth is evaluated before output quantization, just like the WebGL MRT.
struct GenerateUniforms {
    base: u32, count: u32, sourceCount: u32, indexed: u32,
    sourceExt: u32, outputExt: u32, sortRadial: u32, numSh: u32,
    rotation: vec4f,
    translationScale: vec4f,
    viewOrigin: vec4f,
    viewDirection: vec4f, // w: sort cone cos(sortCullFov / 2), -2 when off
    localViewOrigin: vec4f,
    encoding: vec4f,
    shMaxOpacity: vec4f,
    shOffsets: vec4u,
    recolor: vec4f,
// x: SplatEdits.numEdits, y: SplatEdits.numSdfs (sdfData follows editData).
editCounts: vec4u,
// x: SplatMesh.introDots time in seconds, negative when off.
introDots: vec4f,
};
@group(0) @binding(0) var<uniform> frame: GenerateUniforms;
// Bindings 1-3 (and 8-10 for paged sources) and the load*() accessors come
// from generateSourceBuffers.wgsl or generateSourcePaged.wgsl.
@group(0) @binding(4) var<storage, read> indices: array<u32>;
// Written straight into the slot's splat textures (no staging buffer copy).
@group(0) @binding(5) var output: texture_storage_2d_array<rgba32uint, write>;
@group(0) @binding(6) var output2: texture_storage_2d_array<rgba32uint, write>;

fn storeOutput(second: bool, dst: u32, value: vec4u) {
    let size = textureDimensions(output);
    let coord = vec2u(dst % size.x, (dst / size.x) % size.y);
    let layer = dst / (size.x * size.y);
    if (second) { textureStore(output2, coord, layer, value); }
    else { textureStore(output, coord, layer, value); }
}
@group(0) @binding(7) var<storage, read_write> depths: array<u32>;

// Native counterpart of SplatEdit.ts defineSdfArray/defineEdit: world-space
// SDF RGBA/displace layers. editData holds frame.editCounts.x packed edits
// (SplatEdits.editData) followed by 8 vec4u per SDF (SplatEdits.sdfData).
@group(0) @binding(11) var<storage, read> editData: array<vec4u>;
// WGSL rejects infinite constants; FLT_MAX stands in for GLSL 1.0 / 0.0.
const EDIT_INFINITY: f32 = 0x1.fffffep+127f;

fn sdfRow(sdf: u32, row: u32) -> vec4f {
    return bitcast<vec4f>(editData[frame.editCounts.x + sdf * 8u + row]);
}

fn evaluateSdfs(first: u32, count: u32, pos: vec3f, smoothK: f32,
                values: ptr<function, array<vec4f, 2>>) -> f32 {
    var accum = select(0.0, EDIT_INFINITY, smoothK == 0.0);
    var maxExp = -EDIT_INFINITY;
    (*values)[0] = vec4f(0.0);
    (*values)[1] = vec4f(0.0);
    let last = min(first + count, frame.editCounts.y);
    for (var index = first; index < last; index++) {
        let row0 = editData[frame.editCounts.x + index * 8u];
        let flags = row0.w;
        let sizes = sdfRow(index, 3u);
        let sdfValues = array<vec4f, 2>(sdfRow(index, 4u), sdfRow(index, 5u));
        var p = quatVec(sdfRow(index, 1u), pos * sdfRow(index, 2u).xyz) + bitcast<vec3f>(row0.xyz);
        var distance = -EDIT_INFINITY;
        switch (flags & 0xffu) {
            case 1u: { distance = p.z; }
            case 2u: { distance = length(p) - sizes.w; }
            case 3u: {
                let q = abs(p) - sizes.xyz + sizes.w;
                distance = length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - sizes.w;
            }
            case 4u: {
                let k0 = length(p / sizes.xyz);
                let k1 = length(p / dot(sizes.xyz, sizes.xyz));
                distance = k0 * (k0 - 1.0) / k1;
            }
            case 5u: {
                let d = abs(vec2f(length(p.xz), p.y)) - sizes.wy;
                distance = min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0)));
            }
            case 6u: {
                p.y -= clamp(p.y, -0.5 * sizes.y, 0.5 * sizes.y);
                distance = length(p) - sizes.w;
            }
            case 7u: {
                let angle = 0.25 * PI * sizes.w;
                let c = vec2f(sin(angle), cos(angle));
                let q = vec2f(length(p.xy), -p.z);
                let d = length(q - c * max(dot(q, c), 0.0));
                distance = d * select(1.0, -1.0, (q.x * c.y - q.y * c.x) < 0.0);
            }
            default: {}
        }
        if ((flags & 0x100u) != 0u) { distance = -distance; }
        if (smoothK == 0.0) {
            if (distance < accum) {
                accum = distance;
                *values = sdfValues;
            }
        } else {
            let scaled = -distance / smoothK;
            if (scaled > maxExp) {
                let s = exp(maxExp - scaled);
                accum *= s;
                (*values)[0] *= s;
                (*values)[1] *= s;
                maxExp = scaled;
            }
            let weight = exp(scaled - maxExp);
            accum += weight;
            (*values)[0] += weight * sdfValues[0];
            (*values)[1] += weight * sdfValues[1];
        }
    }
    if (smoothK == 0.0) { return accum; }
    // Very distant SDFs may result in 0 accumulation.
    if (accum == 0.0) { return EDIT_INFINITY; }
    (*values)[0] /= accum;
    (*values)[1] /= accum;
    return (-log(accum) - maxExp) * smoothK;
}

fn applyEdits(center: ptr<function, vec3f>, rgba: ptr<function, vec4f>) {
    for (var e = 0u; e < frame.editCounts.x; e++) {
        let packed = editData[e];
        let softEdge = bitcast<f32>(packed.z);
        var values: array<vec4f, 2>;
        var distance = evaluateSdfs(packed.y & 0xffffu, packed.y >> 16u, *center, bitcast<f32>(packed.w), &values);
        if ((packed.x & 0x100u) != 0u) { distance = -distance; }
        var modulate: f32;
        if (softEdge == 0.0) {
            modulate = select(0.0, 1.0, distance < 0.0);
        } else {
            modulate = clamp(-distance / softEdge + 0.5, 0.0, 1.0);
        }
        var blended: vec4f;
        switch (packed.x & 0xffu) {
            case 0u: { blended = *rgba * values[0]; }
            case 1u: { blended = vec4f(values[0].rgb, (*rgba).a * values[0].a); }
            case 2u: { blended = *rgba + values[0]; }
            case 3u: {
                let luma = dot((*rgba).rgb, vec3f(0.2126, 0.7152, 0.0722));
                blended = vec4f(mix(vec3f(luma), (*rgba).rgb, values[0].a), (*rgba).a);
            }
            default: { blended = vec4f(fract(*center), 1.0); }
        }
        *rgba = mix(*rgba, blended, modulate);
        *center += values[1].xyz * modulate;
    }
}

// Native counterpart of SplatMesh.ts introDotsGsplat (smooth value noise).
fn introDotsHash(q: vec3f) -> vec3f {
    let p = fract(q * 0.3183099 + 0.1) * 17.0;
    return fract(vec3f(p.x * p.y * p.z, p.x + p.y * p.z, p.x * p.y + p.z));
}

fn introDotsNoise(p: vec3f) -> vec3f {
    let i = floor(p);
    var f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    let x0 = mix(introDotsHash(i), introDotsHash(i + vec3f(1.0, 0.0, 0.0)), f.x);
    let x1 = mix(introDotsHash(i + vec3f(0.0, 1.0, 0.0)), introDotsHash(i + vec3f(1.0, 1.0, 0.0)), f.x);
    let x2 = mix(introDotsHash(i + vec3f(0.0, 0.0, 1.0)), introDotsHash(i + vec3f(1.0, 0.0, 1.0)), f.x);
    let x3 = mix(introDotsHash(i + vec3f(0.0, 1.0, 1.0)), introDotsHash(i + vec3f(1.0, 1.0, 1.0)), f.x);
    return mix(mix(x0, x1, f.y), mix(x2, x3, f.y), f.z);
}

@compute @workgroup_size(256)
fn generate(@builtin(workgroup_id) group: vec3u,
            @builtin(local_invocation_id) local: vec3u,
            @builtin(num_workgroups) groups: vec3u) {
    let i = (group.y * groups.x + group.x) * 256u + local.x;
    if (i >= frame.count) { return; }
    let dst = frame.base + i;
    var src = i;
    if (frame.indexed != 0u) { src = indices[i]; }
    if (src >= frame.sourceCount) { return; }
    var splat: UnpackedSplat;
    if (frame.sourceExt != 0u) {
        splat = unpackSplatExt(loadPrimary(src), loadSecondary(src));
    } else {
        splat = unpackSplatEncoding(loadPrimary(src), frame.encoding);
        splat.rgba.a *= frame.shMaxOpacity.w;
    }
    // Upstream marks source activity by nonzero scales, not opacity. Zero
    // alpha still owns a finite depth/order entry and is discarded at draw,
    // except for paged sources, whose readers drop it like zero scales.
    if (all(splat.scales == vec3f(0.0)) || (ZERO_ALPHA_INACTIVE && splat.rgba.a == 0.0)) { return; }

    if (frame.numSh > 0u) {
        let dir = normalize(splat.center - frame.localViewOrigin.xyz);
        var rgb = vec3f(0.0);
        if (frame.sourceExt != 0u) {
            let a = loadSh(0u, src);
            if (frame.numSh == 1u) {
                rgb = evaluateExtSH1(a, dir);
            } else {
                rgb = evaluateExtSH12(a, loadSh(1u, src), dir);
                if (frame.numSh >= 3u) {
                    rgb += evaluateExtSH3(loadSh(2u, src), loadSh(3u, src), dir);
                }
            }
        } else {
            rgb = evaluatePackedSH1(loadPackedSh1(src), dir, frame.shMaxOpacity.x);
            if (frame.numSh >= 2u) {
                rgb += evaluatePackedSH2(loadSh(1u, src), dir, frame.shMaxOpacity.y);
            }
            if (frame.numSh >= 3u) {
                rgb += evaluatePackedSH3(loadSh(2u, src), dir, frame.shMaxOpacity.z);
            }
        }
        // Match upstream source generation: signed SH colors survive until
        // output encoding. Extended output preserves negative half floats;
        // packed output performs its own format-specific clamping.
        splat.rgba = vec4f(splat.rgba.rgb + rgb, splat.rgba.a);
    }
    splat.rgba *= frame.recolor;
    // Intro dots in object space, before the mesh transform (as in WebGL).
    if (frame.introDots.x >= 0.0) {
        splat.center += 0.1 * introDotsNoise(splat.center * 2.0 + frame.introDots.x * 0.5);
        splat.scales = vec3f(0.015);
    }
    var center = quatVec(frame.rotation, splat.center * frame.translationScale.w) + frame.translationScale.xyz;
    // SplatMesh applies RGBA/displace edits in world space after recoloring.
    applyEdits(&center, &splat.rgba);
    let delta = center - frame.viewOrigin.xyz;
    let dist = length(delta);
    var metric = dot(delta, frame.viewDirection.xyz) + 100.0;
    if (frame.sortRadial != 0u) { metric = dist; }
    // Negative/NaN/infinite bit patterns are excluded by upstream sort32.
    depths[dst] = bitcast<u32>(metric);
    // Outside the sort cone (sortCullFov) the depth stays +inf: not drawn.
    if (dist > 0.0 && dot(delta, frame.viewDirection.xyz) < frame.viewDirection.w * dist) {
        depths[dst] = 0x7f800000u;
    }
    let scales = splat.scales * frame.translationScale.w;
    let rotation = quatQuat(frame.rotation, splat.quaternion);
    if (frame.outputExt != 0u) {
        let packed = packSplatExt(center, scales, rotation, splat.rgba);
        storeOutput(false, dst, packed.packed);
        storeOutput(true, dst, packed.packed2);
    } else {
        // Upstream uses camera-relative half precision and expanded LoD alpha.
        storeOutput(false, dst, packSplat(delta, scales, rotation, vec4f(splat.rgba.rgb, splat.rgba.a * 0.5)));
    }
}
