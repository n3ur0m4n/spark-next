// splatVertex.wgsl - WGSL port of splatVertex.glsl
// Spark.js 3D Gaussian Splatting renderer - WebGPU vertex shader
//
// This file is self-contained: splatDefines.wgsl content is prepended at build
// time (or concatenated at runtime). For naga validation the defines are
// included inline.

// ============================================================================
// Pipeline-overridable constants
// ============================================================================

override HALVED_ALPHA: bool = true;

// ============================================================================
// Uniform structs and bind group layout
// ============================================================================

struct SparkUniforms {
    renderSize: vec2f,
    renderToViewQuat: vec4f,
    renderToViewPos: vec3f,
    maxStdDev: f32,
    renderToViewBasis: mat3x3f,  // NOTE: mat3x3f has alignment padding in uniform buffer
    minPixelRadius: f32,
    maxPixelRadius: f32,
    enableExtSplats: u32,   // bool as u32 for uniform buffer
    enableCovSplats: u32,   // bool as u32 for uniform buffer
    time: f32,
    deltaTime: f32,
    debugFlag: u32,         // bool as u32 for uniform buffer
    minAlpha: f32,
    enable2DGS: u32,        // bool as u32 for uniform buffer
    blurAmount: f32,
    preBlurAmount: f32,
    focalDistance: f32,
    apertureAngle: f32,
    clipXY: f32,
    focalAdjustment: f32,
    isOrthographic: u32,    // bool as u32 for uniform buffer
    _padding: vec3f,
    lodInflate: u32,
    _padding2: vec4f,
    splatEncoding: vec4f,  // rgbMin, rgbMax, lnScaleMin, lnScaleMax
}

@group(0) @binding(0) var<uniform> uniforms: SparkUniforms;
@group(0) @binding(1) var<uniform> projectionMatrix: mat4x4f;
@group(0) @binding(2) var ordering: texture_2d<u32>;
@group(0) @binding(3) var extSplats: texture_2d_array<u32>;
@group(0) @binding(4) var extSplats2: texture_2d_array<u32>;
// Leading fields of splatFragment.wgsl's FragmentUniforms (same buffer).
struct ColorUniforms {
    near: f32,
    far: f32,
    encodeLinear: u32,
}
@group(0) @binding(5) var<uniform> colorUniforms: ColorUniforms;

// Lines starting with "//PICK " are active only in the GPU pick variant
// (pickVertexWgsl() strips the prefix), mirroring SPARK_PICK_DEPTH in
// splatVertex.glsl: per-fragment depth of the Gaussian's max-density point
// along the pixel ray (see splatPickFragment.wgsl).

// ============================================================================
// Vertex input / output structs
// ============================================================================

struct VertexInput {
    @location(0) position: vec2f,
    @builtin(instance_index) instanceIndex: u32,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) vRgba: vec4f,
    @location(1) vSplatUv: vec2f,
    @location(2) vNdc: vec3f,
    @location(3) @interpolate(flat) vSplatIndex: u32,
    @location(4) @interpolate(flat) adjustedStdDev: f32,
//PICK     @location(5) vPickRay: vec3f,                            // view-space ray dir with z = -1
//PICK     @location(6) @interpolate(flat) vPickCenter: vec3f,      // view-space splat center
//PICK     @location(7) @interpolate(flat) vPickSinvDiag: vec3f,    // inverse(cov3D) up to scale: diagonal
//PICK     @location(8) @interpolate(flat) vPickSinvOff: vec3f,     // ... off-diagonal (xy, xz, yz)
//PICK     @location(9) @interpolate(flat) vPickAux: vec2f,         // x = regularized sqrt(cov zz), y = 1 -> center/NDC depth
}

// ============================================================================
// Helper: offscreen discard position
// ============================================================================

fn discardVertex() -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4f(0.0, 0.0, 2.0, 1.0);
    out.vRgba = vec4f(0.0);
    out.vSplatUv = vec2f(0.0);
    out.vNdc = vec3f(0.0);
    out.vSplatIndex = 0u;
    out.adjustedStdDev = 0.0;
//PICK     out.vPickRay = vec3f(0.0, 0.0, -1.0);
//PICK     out.vPickCenter = vec3f(0.0);
//PICK     out.vPickSinvDiag = vec3f(0.0);
//PICK     out.vPickSinvOff = vec3f(0.0);
//PICK     out.vPickAux = vec2f(0.0, 1.0);
    return out;
}

// ============================================================================
// Vertex shader entry point
// ============================================================================

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    // Default to outside the frustum so it's discarded if we return early
    output.position = vec4f(0.0, 0.0, 2.0, 1.0);
    output.vRgba = vec4f(0.0);
    output.vSplatUv = vec2f(0.0);
    output.vNdc = vec3f(0.0);
    output.vSplatIndex = 0u;
    output.adjustedStdDev = 0.0;
//PICK     output.vPickRay = vec3f(0.0, 0.0, -1.0);
//PICK     output.vPickCenter = vec3f(0.0);
//PICK     output.vPickSinvDiag = vec3f(0.0);
//PICK     output.vPickSinvOff = vec3f(0.0);
//PICK     output.vPickAux = vec2f(0.0, 1.0);

    let instanceIndex = input.instanceIndex;
    let orderingCoord = vec2i(i32((instanceIndex >> 2u) & 4095u), i32(instanceIndex >> 14u));
    let component = instanceIndex & 3u;
    let splatIndexVec = textureLoad(ordering, orderingCoord, 0);

    // Pick the right component (x/y/z/w) based on instance index mod 4
    var splatIndex: u32;
    switch component {
        case 0u: { splatIndex = splatIndexVec.x; }
        case 1u: { splatIndex = splatIndexVec.y; }
        case 2u: { splatIndex = splatIndexVec.z; }
        default: { splatIndex = splatIndexVec.w; }
    }

    if (splatIndex == 0xFFFFFFFFu) {
        // Special value reserved for "no splat"
        return discardVertex();
    }

    let texCoord = splatTexCoord(i32(splatIndex));

    var center: vec3f;
    var scales: vec3f;
    var quaternion: vec4f;
    var rgba: vec4f;
    var cov3D: mat3x3f;
    var xxyyzz: vec3f;
    var xyxzyz: vec3f;
    var zeroScaleX: bool = false;
    var zeroScaleY: bool = false;
    var zeroScaleZ: bool = false;

    if (uniforms.enableExtSplats != 0u) {
        let ext1 = textureLoad(extSplats, texCoord.xy, texCoord.z, 0);
        let alpha = unpackSplatExtAlpha(ext1);
        if (alpha == 0.0 || alpha < uniforms.minAlpha) {
            return discardVertex();
        }
        let ext2 = textureLoad(extSplats2, texCoord.xy, texCoord.z, 0);

        if (uniforms.enableCovSplats == 0u) {
            let unpacked = unpackSplatExt(ext1, ext2);
            center = unpacked.center;
            scales = unpacked.scales;
            quaternion = unpacked.quaternion;
            rgba = unpacked.rgba;
            zeroScaleX = (scales.x == 0.0);
            zeroScaleY = (scales.y == 0.0);
            zeroScaleZ = (scales.z == 0.0);
            if (zeroScaleX && zeroScaleY && zeroScaleZ) {
                return discardVertex();
            }
        } else {
            let unpacked = unpackSplatExtCov(ext1, ext2);
            center = unpacked.center;
            rgba = unpacked.rgba;
            xxyyzz = unpacked.xxyyzz;
            xyxzyz = unpacked.xyxzyz;
            if (xxyyzz.x == 0.0 && xxyyzz.y == 0.0 && xxyyzz.z == 0.0 &&
                xyxzyz.x == 0.0 && xyxzyz.y == 0.0 && xyxzyz.z == 0.0) {
                return discardVertex();
            }
        }
    } else {
        let packed = textureLoad(extSplats, texCoord.xy, texCoord.z, 0);
        if (uniforms.enableCovSplats == 0u) {
            let unpacked = unpackSplatEncoding(packed, uniforms.splatEncoding);
            center = unpacked.center;
            scales = unpacked.scales;
            quaternion = unpacked.quaternion;
            rgba = unpacked.rgba;
            zeroScaleX = (scales.x == 0.0);
            zeroScaleY = (scales.y == 0.0);
            zeroScaleZ = (scales.z == 0.0);
            if (zeroScaleX && zeroScaleY && zeroScaleZ) {
                return discardVertex();
            }
        } else {
            let unpacked = unpackSplatCovEncoding(packed, uniforms.splatEncoding);
            center = unpacked.center;
            rgba = unpacked.rgba;
            xxyyzz = unpacked.xxyyzz;
            xyxzyz = unpacked.xyxzyz;
            if (xxyyzz.x == 0.0 && xxyyzz.y == 0.0 && xxyyzz.z == 0.0 &&
                xyxzyz.x == 0.0 && xyxzyz.y == 0.0 && xyxzyz.z == 0.0) {
                return discardVertex();
            }
        }

        if (HALVED_ALPHA) {
            rgba.a *= 2.0;
        }
        if (rgba.a == 0.0 || rgba.a < uniforms.minAlpha) {
            return discardVertex();
        }
    }

    rgba = vec4f(max(rgba.rgb, vec3f(0.0)), rgba.a);
    // RGB is constant across the quad: convert once per splat, not per fragment.
    if (colorUniforms.encodeLinear != 0u) {
        rgba = vec4f(srgbToLinear(rgba.rgb), rgba.a);
    }
    var adjustedStdDevLocal = uniforms.maxStdDev;
    if (rgba.a > 1.0) {
        // Stretch 1..2 to 1..5
        rgba.a = min(rgba.a * 4.0 - 3.0, 5.0);
        if (uniforms.lodInflate != 0u) {
            let opacity = exp((rgba.a * rgba.a - 1.0) / 2.718281828459045);
            scales *= pow(opacity, 1.0 / 3.0);
            rgba.a = 1.0;
        }
        // Expand the maximum std dev to approximately cover the larger range
        adjustedStdDevLocal = uniforms.maxStdDev + 0.7 * (rgba.a - 1.0);
    }

    // Compute the view space center of the splat
    var viewCenter: vec3f;
    if (uniforms.enableCovSplats == 0u) {
        viewCenter = quatVec(uniforms.renderToViewQuat, center) + uniforms.renderToViewPos;
    } else {
        viewCenter = (uniforms.renderToViewBasis * center) + uniforms.renderToViewPos;
    }

    // Discard splats behind the camera
    if (viewCenter.z >= 0.0) {
        return discardVertex();
    }

    // Compute the clip space center of the splat
    let clipCenter = projectionMatrix * vec4f(viewCenter, 1.0);

    // Discard splats outside near/far planes (WebGPU NDC Z is [0,1])
    if (clipCenter.z < 0.0 || clipCenter.z >= clipCenter.w) {
        return discardVertex();
    }

    // Discard splats more than clipXY times outside the XY frustum
    let clip = uniforms.clipXY * clipCenter.w;
    if (abs(clipCenter.x) > clip || abs(clipCenter.y) > clip) {
        return discardVertex();
    }

    output.vRgba = rgba;
    output.vSplatUv = input.position * adjustedStdDevLocal;

    // Record the splat index for entropy
    output.vSplatIndex = splatIndex;
    output.adjustedStdDev = adjustedStdDevLocal;

    if (uniforms.enableCovSplats == 0u) {
        // Compute view space quaternion of splat
        let viewQuaternion = quatQuat(uniforms.renderToViewQuat, quaternion);

        let anyZeroScale = zeroScaleX || zeroScaleY || zeroScaleZ;
        if (uniforms.enable2DGS != 0u && anyZeroScale) {
            var offset: vec3f;
            if (zeroScaleZ) {
                offset = vec3f(output.vSplatUv.xy * scales.xy, 0.0);
            } else if (zeroScaleY) {
                offset = vec3f(output.vSplatUv.x * scales.x, 0.0, output.vSplatUv.y * scales.z);
            } else {
                offset = vec3f(0.0, output.vSplatUv.xy * scales.yz);
            }

            let viewPos = viewCenter + quatVec(viewQuaternion, offset);
            output.position = projectionMatrix * vec4f(viewPos, 1.0);
            output.vNdc = output.position.xyz / output.position.w;
//PICK             // Real quad geometry: vNdc depth is already exact -> fallback.
//PICK             output.vPickCenter = viewCenter;
            return output;
        }

        // Compute the 3D covariance matrix of the splat
        let RS = scaleQuaternionToMatrix(scales, viewQuaternion);
        cov3D = RS * transpose(RS);
    } else {
        cov3D = mat3x3f(
            vec3f(xxyyzz.x, xyxzyz.x, xyxzyz.y),
            vec3f(xyxzyz.x, xxyyzz.y, xyxzyz.z),
            vec3f(xyxzyz.y, xyxzyz.z, xxyyzz.z)
        );
        cov3D = uniforms.renderToViewBasis * cov3D * transpose(uniforms.renderToViewBasis);
    }

    // Compute the Jacobian of the splat's projection at its center
    let scaledRenderSize = uniforms.renderSize * uniforms.focalAdjustment;
    let focal = 0.5 * scaledRenderSize * vec2f(projectionMatrix[0][0], projectionMatrix[1][1]);

    // NDC center of the splat
    let ndcCenter = clipCenter.xyz / clipCenter.w;

//PICK     // Cheap pre-test: the pick render covers exactly NDC [-1,1]^2 and a quad
//PICK     // half-extent is at most sqrt(2) * maxPixelRadius pixels.
//PICK     let maxExtNdc = (2.0 * 1.4142136 * uniforms.maxPixelRadius) / scaledRenderSize;
//PICK     if (any(abs(ndcCenter.xy) > vec2f(1.0) + maxExtNdc)) {
//PICK         return discardVertex();
//PICK     }

    var J: mat3x3f;
    if (uniforms.isOrthographic != 0u) {
        J = mat3x3f(
            vec3f(focal.x, 0.0, 0.0),
            vec3f(0.0, focal.y, 0.0),
            vec3f(0.0, 0.0, 0.0)
        );
    } else {
        let invZ = 1.0 / viewCenter.z;
        let J1 = focal * invZ;
        let J2 = -(J1 * viewCenter.xy) * invZ;
        J = mat3x3f(
            vec3f(J1.x, 0.0, J2.x),
            vec3f(0.0, J1.y, J2.y),
            vec3f(0.0, 0.0, 0.0)
        );
    }

    // Compute the 2D covariance by projecting the 3D covariance
    // and picking out the XY plane components.
    let cov2D = transpose(J) * cov3D * J;
    var a = cov2D[0][0];
    var d = cov2D[1][1];
    var b = cov2D[0][1];

    // Optionally pre-blur the splat to match non-antialias optimized splats
    a += uniforms.preBlurAmount;
    d += uniforms.preBlurAmount;

    var fullBlurAmount = uniforms.blurAmount;
    if (uniforms.focalDistance > 0.0 && uniforms.apertureAngle > 0.0) {
        var focusRadius = uniforms.maxPixelRadius;
        if (viewCenter.z < 0.0) {
            let focusBlur = abs((-viewCenter.z - uniforms.focalDistance) / viewCenter.z);
            let apertureRadius = focal.x * tan(0.5 * uniforms.apertureAngle);
            focusRadius = focusBlur * apertureRadius;
        }
        fullBlurAmount = clamp(sqr(focusRadius), uniforms.blurAmount, sqr(uniforms.maxPixelRadius));
    }

    // Do convolution with a 0.5-pixel Gaussian for anti-aliasing: sqrt(0.3) ~= 0.5
    let detOrig = a * d - b * b;
    a += fullBlurAmount;
    d += fullBlurAmount;
    let det = a * d - b * b;

    // Compute anti-aliasing intensity scaling factor
    let blurAdjust = sqrt(max(0.0, detOrig / det));
    rgba.a *= blurAdjust;
    if (rgba.a < uniforms.minAlpha) {
        return discardVertex();
    }
    output.vRgba.a = rgba.a;

    // Compute the eigenvalue and eigenvectors of the 2D covariance matrix
    // Mirrors splatVertex.glsl: cancellation-free eigen decomposition.
    let eigenAvg = 0.5 * (a + d);
    let halfDiagDiff = 0.5 * (a - d);
    let eigenDelta = sqrt(sqr(halfDiagDiff) + sqr(b));
    let eigen1 = eigenAvg + eigenDelta;
    let eigen2 = max(0.0, det / eigen1);

    let eigenVec1Unnorm = select(
        vec2f(b, eigenDelta - halfDiagDiff),
        vec2f(eigenDelta + halfDiagDiff, b),
        halfDiagDiff >= 0.0,
    );
    let eigenVec1Len2 = dot(eigenVec1Unnorm, eigenVec1Unnorm);
    let eigenVec1 = select(
        vec2f(1.0, 0.0),
        eigenVec1Unnorm * inverseSqrt(eigenVec1Len2),
        eigenVec1Len2 > 0.0,
    );
    let eigenVec2 = vec2f(eigenVec1.y, -eigenVec1.x);

    let scale1 = min(uniforms.maxPixelRadius, adjustedStdDevLocal * sqrt(eigen1));
    let scale2 = min(uniforms.maxPixelRadius, adjustedStdDevLocal * sqrt(eigen2));
    if (scale1 < uniforms.minPixelRadius && scale2 < uniforms.minPixelRadius) {
        return discardVertex();
    }

//PICK     // Exact early-out: bounds of the full quad vs the pick rectangle.
//PICK     let extNdc = (2.0 / scaledRenderSize) * (abs(eigenVec1) * scale1 + abs(eigenVec2) * scale2);
//PICK     if (any(ndcCenter.xy - extNdc > vec2f(1.0)) || any(ndcCenter.xy + extNdc < vec2f(-1.0))) {
//PICK         return discardVertex();
//PICK     }
//PICK     {
//PICK         // Same normalization/regularization as splatVertex.glsl; t* is
//PICK         // invariant to the uniform scale of S, so it is never undone.
//PICK         let covMax = max(max(cov3D[0][0], cov3D[1][1]), max(cov3D[2][2], 1e-30));
//PICK         let epsRel = 1e-6;
//PICK         let c = cov3D * (1.0 / covMax) + mat3x3f(vec3f(epsRel, 0.0, 0.0), vec3f(0.0, epsRel, 0.0), vec3f(0.0, 0.0, epsRel));
//PICK         // Symmetric 3x3 inverse via the adjugate (WGSL has no inverse()).
//PICK         let i00 = c[1][1] * c[2][2] - c[1][2] * c[2][1];
//PICK         let i01 = c[0][2] * c[2][1] - c[0][1] * c[2][2];
//PICK         let i02 = c[0][1] * c[1][2] - c[0][2] * c[1][1];
//PICK         let invDet = 1.0 / (c[0][0] * i00 + c[1][0] * i01 + c[2][0] * i02);
//PICK         let i11 = c[0][0] * c[2][2] - c[0][2] * c[2][0];
//PICK         let i12 = c[0][2] * c[1][0] - c[0][0] * c[1][2];
//PICK         let i22 = c[0][0] * c[1][1] - c[0][1] * c[1][0];
//PICK         output.vPickSinvDiag = vec3f(i00, i11, i22) * invDet;
//PICK         output.vPickSinvOff = vec3f(i01, i02, i12) * invDet;
//PICK         output.vPickCenter = viewCenter;
//PICK         output.vPickAux = vec2f(sqrt(cov3D[2][2] + epsRel * covMax), select(0.0, 1.0, uniforms.isOrthographic != 0u));
//PICK     }

    // Compute the NDC coordinates for the ellipsoid's diagonal axes.
    let pixelOffset = input.position.x * eigenVec1 * scale1 + input.position.y * eigenVec2 * scale2;
    let ndcOffset = (2.0 / scaledRenderSize) * pixelOffset;
    let ndc = vec3f(ndcCenter.xy + ndcOffset, ndcCenter.z);

    output.vNdc = ndc;
    output.position = vec4f(ndc.xy * clipCenter.w, clipCenter.zw);
//PICK     // setViewOffset puts its crop offset in P[2][0], P[2][1]; w is constant
//PICK     // over the quad, so the varying interpolates exactly.
//PICK     output.vPickRay = vec3f(
//PICK         (ndc.x + projectionMatrix[2][0]) / projectionMatrix[0][0],
//PICK         (ndc.y + projectionMatrix[2][1]) / projectionMatrix[1][1],
//PICK         -1.0,
//PICK     );

    return output;
}
