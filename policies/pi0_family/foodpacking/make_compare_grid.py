# SPDX-License-Identifier: Apache-2.0
"""4-column comparison grid: {zero-shot, fine-tune} x {reconstruction, GT}, rows = eps 4, 9, 12, 0.

Cells are cropped from the 5x4 tiled 20-env eval videos written by eval.py --video (env i == DR
episode i). Scores and per-episode outcomes below are typed in from the results.json files; edit
COLS when re-running. Runs in the openpi venv (needs av + Pillow), no Isaac:
    openpi/.venv/bin/python policies/pi0_family/foodpacking/make_compare_grid.py
"""
import argparse, av, numpy as np, os
from PIL import Image, ImageDraw, ImageFont
V = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "videos")
ap = argparse.ArgumentParser()
ap.add_argument("--eps", default="4,9,12,0", help="episode rows, comma-separated")
ap.add_argument("--cell", type=int, default=400, help="cell width px (16:9)")
ap.add_argument("--panel", type=int, default=1, help="1 = right-hand context panel, 0 = compact footer instead")
ap.add_argument("--out", default="compare_grid_final.mp4", help="file name under videos/")
A = ap.parse_args()
OUT = f"{V}/{A.out}"
COLS = [  # (video, title, scene, mustard/20, spam/20, full/20, per-episode outcomes {ep: (mustard, spam, solved_step)})
    (f"{V}/zs_vec.mp4",      "ZERO-SHOT pi05 (DROID)", "reconstructed scene", 0, 0, 0,  {4: (0, 0, None), 9: (0, 0, None), 12: (0, 0, None), 0: (0, 0, None)}),
    (f"{V}/ft_vec.mp4",      "FINE-TUNED LoRA, 123 demos", "reconstructed scene", 12, 5, 5, {4: (1, 1, 379), 9: (1, 1, 520), 12: (1, 1, 345), 0: (0, 0, None)}),
    (f"{V}/gt_zeroshot.mp4", "ZERO-SHOT pi05 (DROID)", "RoboLab GT scene", 1, 0, 0,     {4: (0, 0, None), 9: (0, 0, None), 12: (0, 0, None), 0: (0, 0, None)}),
    (f"{V}/gt_finetune.mp4", "FINE-TUNED LoRA (same ckpt)", "RoboLab GT scene", 11, 0, 0, {4: (1, 0, None), 9: (1, 0, None), 12: (1, 0, None), 0: (0, 0, None)}),
]
EPS = [int(e) for e in A.eps.split(",")]
CW = A.cell; CH = CW * 9 // 16            # cell size (320x180 source, scaled)
GAP, LBL, HDR, PAD = 12, 30, 118, 14
PANEL = 520 if A.panel else 0
FOOT = 0 if A.panel else 58
W = PAD + 4 * (CW + GAP) + PANEL
H = HDR + len(EPS) * (LBL + CH + GAP) + FOOT + PAD
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans"   # Debian/Ubuntu fonts-dejavu
F = lambda s, b=False: ImageFont.truetype(f"{FONT}{'-Bold' if b else ''}.ttf", s)
GREEN, RED, GREY, WHITE, DIM = (70, 230, 110), (240, 90, 80), (150, 150, 150), (235, 235, 235), (120, 120, 120)

def cell_xy(c, r):
    return PAD + c * (CW + GAP), HDR + r * (LBL + CH + GAP)

# ---- static layer
bg = Image.new("RGB", (W, H), (10, 10, 10)); d = ImageDraw.Draw(bg)
for c, (_, title, scene, mu, sp, full, _) in enumerate(COLS):
    x, _ = cell_xy(c, 0)
    d.text((x, 10), title, font=F(19, True), fill=WHITE)
    d.text((x, 36), scene, font=F(16), fill=(255, 210, 90) if "GT" in scene else (140, 190, 255))
    d.text((x, 62), f"mustard {mu}/20", font=F(22, True), fill=GREEN if mu else GREY)
    d.text((x + 190, 62), f"spam {sp}/20", font=F(22, True), fill=GREEN if sp else GREY)
    d.text((x, 92), f"full task {full}/20", font=F(15), fill=DIM)
for r, ep in enumerate(EPS):
    for c, (_, _, _, _, _, _, outc) in enumerate(COLS):
        x, y = cell_xy(c, r); m, s, solved = outc[ep]
        d.text((x, y + 6), f"ep {ep}", font=F(15), fill=WHITE)
        d.text((x + 60, y + 6), "mustard " + ("IN" if m else "out"), font=F(15, True if m else False), fill=GREEN if m else GREY)
        d.text((x + 185, y + 6), "spam " + ("IN" if s else "out"), font=F(15, True if s else False), fill=GREEN if s else GREY)
        if solved:
            d.text((x + 290, y + 6), f"FULL@{solved}", font=F(15, True), fill=GREEN)
# panel
def block(y, head, lines, gap=24):
    d.text((px, y), head, font=F(20, True), fill=WHITE); y += 32
    for ln in lines:
        col = WHITE
        if isinstance(ln, tuple): ln, col = ln
        d.text((px, y), ln, font=F(15), fill=col); y += gap
    return y + 14
if not A.panel:
    fy = HDR + len(EPS) * (LBL + CH + GAP) + 4
    d.text((PAD, fy), 'prompt: "put the mustard bottle in the left bin and the spam can in the right bin"   |   '
           "fine-tune demos recorded on the RECONSTRUCTED scene only; GT columns = transfer test", font=F(15), fill=WHITE)
    d.text((PAD, fy + 24), "sub-score = object inside its bin at any time (measured footprint, below rim); 20 DR episodes, "
           f"same start poses in all columns; shown: ep {', '.join(map(str, EPS))}; real time 15 Hz", font=F(15), fill=DIM)
px = PAD + 4 * (CW + GAP) + 8
y = 14
if A.panel:
  d.line([(px - 8, 8), (px - 8, H - 8)], fill=(60, 60, 60), width=2)
  y = block(y, "TASK", ['prompt: "put the mustard bottle in the left bin', '   and the spam can in the right bin"',
                        "robot: Franka + Robotiq 2F-85, Isaac Lab / PhysX",
                        "DR: mustard + spam XY +/-4 cm, same 20 held-out draws",
                        "start pose: per episode, identical across all 4 columns"])
  y = block(y, "SCENES", [("reconstructed: Robolab delivery-adapter rebuild of the", (140, 190, 255)),
                          ("   RoboLab food_packing scene (recon meshes + textures)", (140, 190, 255)),
                          ("GT: RoboLab's authored food_packing.usda (ycb + vomp)", (255, 210, 90)),
                          "only table + objects differ; robot, cameras, physics,",
                          "DR draws, start poses and scoring are identical"])
  y = block(y, "POLICIES", ["zero-shot: pi05_droid_jointpos, no adaptation",
                            "fine-tuned: same base + LoRA on 123 self-generated",
                            "   expert demos, 3000 steps -- demos recorded on the",
                            "   RECONSTRUCTED scene only; GT column = transfer test",
                            "obs: exterior + wrist RGB (224 pad), 7 joints, gripper"])
  y = block(y, "SCORING (sub-scores)", ["per object: inside its bin at any time (latched),",
                                        "   measured inner footprint, below rim; 600-step cap",
                                        "full task = both objects; sub-scores are the signal here"])
  y = block(y, "RESULT (20 episodes)", [
      ("                        mustard   spam   full", DIM),
      ("zero-shot   recon          0/20    0/20   0/20", GREY),
      ("fine-tuned  recon         12/20    5/20   5/20", GREEN),
      ("zero-shot   GT             1/20    0/20   0/20", GREY),
      ("fine-tuned  GT            11/20    0/20   0/20", (255, 210, 90)),
      "", "mustard transfers recon -> GT (11 vs 12);",
      "spam does not (0 vs 5): touched in 7/20, never lifted",
      "", ("shown: eps 4, 9, 12 (fine-tune scores) + ep 0 (all fail)", DIM),
      ("real time, 15 Hz", DIM)], gap=22)
static = np.array(bg)

# ---- stream
ins = [av.open(v) for v, *_ in COLS]
gens = [c.decode(video=0) for c in ins]
out = av.open(OUT, "w"); st = out.add_stream("libx264", rate=15); st.width, st.height = W, H; st.pix_fmt = "yuv420p"; st.options = {"crf": "22"}
n = 0
while True:
    try:
        frames = [next(g).to_ndarray(format="rgb24") for g in gens]
    except StopIteration:
        break
    img = static.copy()
    for c, fr in enumerate(frames):
        for r, ep in enumerate(EPS):
            rr, cc = divmod(ep, 5)
            cell = fr[rr * 180:(rr + 1) * 180, cc * 320:(cc + 1) * 320]
            cell = np.array(Image.fromarray(cell).resize((CW, CH), Image.BILINEAR))
            x, y0 = cell_xy(c, r); img[y0 + LBL:y0 + LBL + CH, x:x + CW] = cell
    pil = Image.fromarray(img); dd = ImageDraw.Draw(pil)
    dd.text((PAD, H - PAD - 18) if A.panel else (W - PAD - 90, HDR + len(EPS) * (LBL + CH + GAP) + 4), f"step {n}", font=F(15), fill=DIM)
    for pkt in st.encode(av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")):
        out.mux(pkt)
    n += 1
for pkt in st.encode():
    out.mux(pkt)
out.close(); [c.close() for c in ins]
print("frames", n, "size", W, "x", H, "->", OUT, os.path.getsize(OUT) // 1024, "KB")
