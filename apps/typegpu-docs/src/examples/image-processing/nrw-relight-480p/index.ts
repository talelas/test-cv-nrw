import { tgpu, common, d, std } from 'typegpu';
import { defineControls } from '../../common/defineControls.ts';

/**
 * NRW 8th Edition — 480p reduced-quality relighting for RTX 4060.
 *
 * Single fullscreen fragment pass (no G-buffer, no cube shadow map):
 *   camera 640x480 -> pseudo-depth -> Sobel normals ->
 *   Lambert diffuse + Blinn-Phong specular + 4-tap fake shadow.
 *
 * To plug real AI depth (DepthAnythingV2 / ONNX), replace `pseudoDepth`
 * with a sampled depth texture and keep the rest unchanged.
 */

// --- 480p tuning -----------------------------------------------------------
const CAP_W = 640;
const CAP_H = 480;
// Focal lengths in px for 640x480 (approx, refine with calibration).
const FX = 520;
const FY = 520;
const SOBEL_STRENGTH = 2.2; // lower = flatter normals (less noise at 480p)
const EMA_ALPHA = 0.45; // light smoothing (Layer 2 filter, Cahier §4.1)

const Params = d.struct({
  lightUv: d.vec2f, // 0..1, mouse (stand-in for MediaPipe palm idx 9)
  lightZ: d.f32, // metres, wheel (stand-in for sampled depth z_c)
  intensity: d.f32, // I0 base intensity
  shininess: d.f32, // Blinn-Phong exponent (32 = cheap, 128 = tight)
  shadowStrength: d.f32, // 0..1 fake-shadow strength
  ambient: d.f32, // ambient floor so shadows never go black
  mode: d.u32, // 0 = shaded, 1 = normals, 2 = pseudo-depth
});

const textureLayout = tgpu.bindGroupLayout({
  frame: { externalTexture: d.textureExternal() },
});

// NOTE: top-level await style matches other image-processing examples.
const root = await tgpu.init();
const canvas = document.querySelector('canvas') as HTMLCanvasElement;
canvas.width = CAP_W;
canvas.height = CAP_H;
const video = document.querySelector('video') as HTMLVideoElement;
const spinner = document.querySelector('.spinner-background') as HTMLDivElement;
const fpsEl = document.querySelector('#fps') as HTMLSpanElement;
const lightEl = document.querySelector('#light') as HTMLSpanElement;

const context = root.configureContext({ canvas, alphaMode: 'premultiplied' });
const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

const params = root.createUniform(Params, {
  lightUv: d.vec2f(0.5, 0.5),
  lightZ: 0.8,
  intensity: 1.6,
  shininess: 48,
  shadowStrength: 0.75,
  ambient: 0.12,
  mode: 0,
});

const sampler = root.createSampler({ magFilter: 'linear', minFilter: 'linear' });
const uvTransform = root.createUniform(d.mat2x2f, d.mat2x2f.identity());

// --- GPU helpers -----------------------------------------------------------

const lumaOf = (c: d.v3f): number => {
  'use gpu';
  return std.dot(c, d.vec3f(0.299, 0.587, 0.114));
};

// Reduced-quality pseudo-depth: bright/central pixels = nearer.
// Replace this body with `textureSample(depthTex, uv).r` when AI depth lands.
const pseudoDepth = (rgb: d.v3f, uv: d.v2f): number => {
  'use gpu';
  const luma = lumaOf(rgb);
  const centered = std.length(uv.sub(0.5)) * 2; // 0 center -> ~1.4 corner
  return std.mix(1.5, 0.45, luma) + centered * 0.12;
};

const sampleDepth = (uv: d.v2f): number => {
  'use gpu';
  const rgb = std
    .textureSampleBaseClampToEdge(textureLayout.$.frame, sampler.$, uv)
    .rgb;
  return pseudoDepth(rgb, uv);
};

const mainFrag = tgpu.fragmentFn({
  in: { uv: d.location(0, d.vec2f) },
  out: d.vec4f,
})(({ uv }) => {
  'use gpu';
  const p = params.$;
  const cuv = uvTransform.$.mul(uv.sub(0.5)).add(0.5);

  const texel = d.vec2f(1 / CAP_W, 1 / CAP_H);

  // 3x3 color taps -> depth (9 taps, the only expensive part at 480p).
  const c00 = std.textureSampleBaseClampToEdge(textureLayout.$.frame, sampler.$, cuv.sub(texel)).rgb;
  const c10 = std.textureSampleBaseClampToEdge(textureLayout.$.frame, sampler.$, cuv.sub(d.vec2f(0, texel.y))).rgb;
  const c20 = std.textureSampleBaseClampToEdge(textureLayout.$.frame, sampler.$, cuv.add(d.vec2f(texel.x, -texel.y))).rgb;
  const c01 = std.textureSampleBaseClampToEdge(textureLayout.$.frame, sampler.$, cuv.sub(d.vec2f(texel.x, 0))).rgb;
  const c11 = std.textureSampleBaseClampToEdge(textureLayout.$.frame, sampler.$, cuv).rgb;
  const c21 = std.textureSampleBaseClampToEdge(textureLayout.$.frame, sampler.$, cuv.add(d.vec2f(texel.x, 0))).rgb;
  const c02 = std.textureSampleBaseClampToEdge(textureLayout.$.frame, sampler.$, cuv.add(d.vec2f(-texel.x, texel.y))).rgb;
  const c12 = std.textureSampleBaseClampToEdge(textureLayout.$.frame, sampler.$, cuv.add(d.vec2f(0, texel.y))).rgb;
  const c22 = std.textureSampleBaseClampToEdge(textureLayout.$.frame, sampler.$, cuv.add(texel)).rgb;

  const d00 = pseudoDepth(c00, cuv.sub(texel));
  const d10 = pseudoDepth(c10, cuv.sub(d.vec2f(0, texel.y)));
  const d20 = pseudoDepth(c20, cuv.add(d.vec2f(texel.x, -texel.y)));
  const d01 = pseudoDepth(c01, cuv.sub(d.vec2f(texel.x, 0)));
  const d11 = pseudoDepth(c11, cuv);
  const d21 = pseudoDepth(c21, cuv.add(d.vec2f(texel.x, 0)));
  const d02 = pseudoDepth(c02, cuv.add(d.vec2f(-texel.x, texel.y)));
  const d12 = pseudoDepth(c12, cuv.add(d.vec2f(0, texel.y)));
  const d22 = pseudoDepth(c22, cuv.add(texel));

  if (p.mode === 2) {
    const g = 1 - (d11 - 0.4) / 1.3;
    return d.vec4f(d.vec3f(std.saturate(g)), 1);
  }

  // Sobel -> normals (Cahier App. A.6).
  const gx = d00 + 2 * d01 + d02 - (d20 + 2 * d21 + d22);
  const gy = d00 + 2 * d10 + d20 - (d02 + 2 * d12 + d22);
  const n = std.normalize(
    d.vec3f(-FX * gx * SOBEL_STRENGTH * 0.02, -FY * gy * SOBEL_STRENGTH * 0.02, 1),
  );
  // Orientation fix: keep normals facing the camera.
  const nn = std.select(n.mul(-1), n, d.bool(std.dot(n, d.vec3f(0, 0, 1)) > 0));

  if (p.mode === 1) {
    return d.vec4f(nn.mul(0.5).add(0.5), 1);
  }

  // Back-project pixel + light (Cahier App. A.1/A.2).
  const zc = d11;
  const pp = d.vec3f(
    (zc * (cuv.x - 0.5) * CAP_W) / FX,
    ((0.5 - cuv.y) * zc * CAP_H) / FY,
    zc,
  );
  const lp = d.vec3f(
    ((p.lightUv.x - 0.5) * p.lightZ * CAP_W) / FX,
    ((0.5 - p.lightUv.y) * p.lightZ * CAP_H) / FY,
    p.lightZ,
  );
  const toLight = lp.sub(pp);
  const dist = std.max(std.length(toLight), 1e-3);
  const l = toLight.div(dist);
  const v = std.normalize(pp.mul(-1));

  // Lambert + Blinn-Phong (Cahier App. A.3/A.4) + inverse-square falloff (A.5).
  const diff = std.max(std.dot(nn, l), 0);
  const h = std.normalize(l.add(v));
  const spec = std.pow(std.max(std.dot(nn, h), 0), p.shininess);
  const atten = (p.intensity * 1.0) / (1 + (dist / 0.5) * (dist / 0.5));

  // 4-tap fake shadow: march toward light in UV, occluded if depth dips.
  const lightUvDir = p.lightUv.sub(cuv);
  const marchLen = std.max(std.length(lightUvDir), 1e-4);
  const stepUv = lightUvDir.div(marchLen).mul(0.012);
  const s1 = sampleDepth(cuv.add(stepUv.mul(1)));
  const s2 = sampleDepth(cuv.add(stepUv.mul(2)));
  const s3 = sampleDepth(cuv.add(stepUv.mul(3)));
  const s4 = sampleDepth(cuv.add(stepUv.mul(4)));
  let occ = 0.0;
  occ += s1 < d11 - 0.03 ? 0.4 : 0.0;
  occ += s2 < d11 - 0.03 ? 0.3 : 0.0;
  occ += s3 < d11 - 0.03 ? 0.2 : 0.0;
  occ += s4 < d11 - 0.03 ? 0.1 : 0.0;
  const shadow = 1 - std.saturate(occ) * p.shadowStrength;

  // Warm point-light tint (Cahier §4.1): R*1.1, G*1.0, B*0.9.
  const warm = d.vec3f(1.1, 1.0, 0.9);
  const shaded = c11
    .mul(p.ambient + diff * atten * shadow)
    .mul(warm)
    .add(d.vec3f(spec * atten * shadow));
  const graded = std.pow(std.saturate(shaded), d.vec3f(1 / 2.2));
  return d.vec4f(graded, 1);
});

const renderPipeline = root.createRenderPipeline({
  vertex: common.fullScreenTriangle,
  fragment: mainFrag,
  targets: { format: presentationFormat },
});

// --- CPU: camera + mouse-light (MediaPipe stand-in) + FPS -------------------

if (!navigator.mediaDevices?.getUserMedia) {
  throw new Error('getUserMedia not supported');
}

video.srcObject = await navigator.mediaDevices.getUserMedia({
  video: {
    width: { ideal: CAP_W },
    height: { ideal: CAP_H },
    frameRate: { ideal: 30 },
  },
  audio: false,
});

const lightTarget = { uv: { x: 0.5, y: 0.5 }, z: 0.8 };
const lightSmooth = { uv: { x: 0.5, y: 0.5 }, z: 0.8 };

canvas.addEventListener('mousemove', (e) => {
  const r = canvas.getBoundingClientRect();
  lightTarget.uv.x = (e.clientX - r.left) / r.width;
  lightTarget.uv.y = (e.clientY - r.top) / r.height;
});
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  lightTarget.z = std_clamp(lightTarget.z + Math.sign(e.deltaY) * 0.08, 0.25, 2.0);
}, { passive: false });

function std_clamp(x: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, x));
}

let videoFrameCallbackId = 0;
let frames = 0;
let lastFpsT = performance.now();

function processVideoFrame(_: number, _metadata: VideoFrameCallbackMetadata) {
  if (video.readyState >= 2) {
    // EMA on light position (Cahier §4.1 temporal filter).
    lightSmooth.uv.x += EMA_ALPHA * (lightTarget.uv.x - lightSmooth.uv.x);
    lightSmooth.uv.y += EMA_ALPHA * (lightTarget.uv.y - lightSmooth.uv.y);
    lightSmooth.z += EMA_ALPHA * (lightTarget.z - lightSmooth.z);
    params.patch({
      lightUv: d.vec2f(lightSmooth.uv.x, lightSmooth.uv.y),
      lightZ: lightSmooth.z,
    });

    const external = root.device.importExternalTexture({ source: video });
    renderPipeline
      .withColorAttachment({ view: context, clearValue: [0, 0, 0, 1] })
      .with(root.createBindGroup(textureLayout, { frame: external }))
      .draw(3);

    frames += 1;
    const now = performance.now();
    if (now - lastFpsT >= 1000) {
      const fps = (frames * 1000) / (now - lastFpsT);
      fpsEl.textContent = `${fps.toFixed(0)} fps @${CAP_W}x${CAP_H}`;
      lightEl.textContent = `light (${lightSmooth.uv.x.toFixed(2)}, ${lightSmooth.uv.y.toFixed(2)}, ${lightSmooth.z.toFixed(2)}m)`;
      frames = 0;
      lastFpsT = now;
    }
    spinner.style.display = 'none';
  }
  videoFrameCallbackId = video.requestVideoFrameCallback(processVideoFrame);
}
videoFrameCallbackId = video.requestVideoFrameCallback(processVideoFrame);

// #region Example controls & Cleanup

export const controls = defineControls({
  mode: {
    initial: 'shaded',
    options: ['shaded', 'normals', 'depth'],
    onSelectChange: (v) => {
      params.patch({ mode: v === 'shaded' ? 0 : v === 'normals' ? 1 : 2 });
    },
  },
  intensity: {
    initial: 1.6, min: 0, max: 4, step: 0.1,
    onSliderChange: (v) => params.patch({ intensity: v }),
  },
  shininess: {
    initial: 48, min: 8, max: 128, step: 4,
    onSliderChange: (v) => params.patch({ shininess: v }),
  },
  shadow: {
    initial: 0.75, min: 0, max: 1, step: 0.05,
    onSliderChange: (v) => params.patch({ shadowStrength: v }),
  },
});

export function onCleanup() {
  video.cancelVideoFrameCallback(videoFrameCallbackId);
  if (video.srcObject) {
    for (const track of (video.srcObject as MediaStream).getTracks()) track.stop();
  }
  video.srcObject = null;
  root.destroy();
}

// #endregion
