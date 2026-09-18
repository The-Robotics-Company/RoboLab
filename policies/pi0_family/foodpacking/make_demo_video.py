# SPDX-License-Identifier: Apache-2.0
"""Side-by-side video of a few self-generated training demos (exterior over wrist), straight from
the ep*.npz files gen_demos.py writes, at the 320x180 the policy is trained on (upscaled).
    openpi/.venv/bin/python make_demo_video.py --demos ~/Desktop/trc/datasets/food_packing_demos_v3 --eps 20,100,179 --out training_demos.mp4
"""
import argparse, os
import av, numpy as np
from PIL import Image, ImageDraw, ImageFont
ap = argparse.ArgumentParser()
ap.add_argument("--demos", required=True)
ap.add_argument("--eps", default="20,100,179")
ap.add_argument("--cell", type=int, default=480)
ap.add_argument("--out", required=True)
ap.add_argument("--title", default="Self-generated training demos -- RMPflow expert, reconstructed scene, legacy wrist cam")
A = ap.parse_args()
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans"
F = lambda s, b=False: ImageFont.truetype(f"{FONT}{'-Bold' if b else ''}.ttf", s)
WHITE, DIM, GREEN = (235, 235, 235), (130, 130, 130), (70, 230, 110)
eps = [int(e) for e in A.eps.split(",")]
def load(e):   # materialise: indexing an NpzFile re-decompresses the whole array every access
    with np.load(os.path.join(A.demos, f"ep{e:04d}.npz")) as z:
        return {k: z[k] for k in ("exterior_image", "wrist_image", "actions", "prompt")}
D = [load(e) for e in eps]
T = max(len(d["exterior_image"]) for d in D)
CW, CH = A.cell, A.cell * 9 // 16
GAP, PAD, HDR, LBL, FOOT = 12, 14, 64, 26, 46
W = PAD * 2 + len(eps) * (CW + GAP) - GAP
H = HDR + LBL + CH + 6 + CH + FOOT + PAD
bg = Image.new("RGB", (W, H), (10, 10, 10)); d = ImageDraw.Draw(bg)
d.text((PAD, 10), A.title, font=F(20, True), fill=WHITE)
d.text((PAD, 38), "top: exterior camera   bottom: wrist camera   -- both 320x180 as fed to pi0.5 (upscaled here)", font=F(14), fill=DIM)
for i, (e, dd) in enumerate(zip(eps, D)):
    x = PAD + i * (CW + GAP)
    d.text((x, HDR + 4), f"demo ep {e}   {len(dd['exterior_image'])} steps @ 15 Hz   {str(dd['prompt'])[:60]}", font=F(14), fill=WHITE)
d.text((PAD, H - FOOT - PAD + 8), "actions: 15-step absolute joint-position chunks + gripper; DR: mustard + spam XY +/-4 cm per episode; "
       "kept 153 demos, 123 used for the LoRA fine-tune", font=F(14), fill=DIM)
static = np.array(bg)
out = av.open(A.out, "w"); st = out.add_stream("libx264", rate=15); st.width, st.height = W, H; st.pix_fmt = "yuv420p"; st.options = {"crf": "22"}
for t in range(T):
    img = static.copy()
    for i, dd in enumerate(D):
        k = min(t, len(dd["exterior_image"]) - 1)
        x = PAD + i * (CW + GAP); y = HDR + LBL
        img[y:y + CH, x:x + CW] = np.array(Image.fromarray(dd["exterior_image"][k]).resize((CW, CH), Image.BILINEAR))
        img[y + CH + 6:y + 2 * CH + 6, x:x + CW] = np.array(Image.fromarray(dd["wrist_image"][k]).resize((CW, CH), Image.BILINEAR))
    pil = Image.fromarray(img); dr = ImageDraw.Draw(pil)
    for i, dd in enumerate(D):
        k = min(t, len(dd["exterior_image"]) - 1); x = PAD + i * (CW + GAP)
        g = float(dd["actions"][k, 7]) > 0.5
        dr.text((x + CW - 150, HDR + LBL + 6), f"step {k}  grip {'CLOSED' if g else 'open'}", font=F(13, g), fill=GREEN if g else DIM)
        if k == len(dd["exterior_image"]) - 1 and t > k:
            dr.text((x + 8, HDR + LBL + 6), "done", font=F(13, True), fill=GREEN)
    for pkt in st.encode(av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")):
        out.mux(pkt)
for pkt in st.encode():
    out.mux(pkt)
out.close(); print("frames", T, "size", W, "x", H, "->", A.out, os.path.getsize(A.out) // 1024, "KB")
