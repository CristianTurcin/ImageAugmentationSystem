from __future__ import annotations
import os, re, glob, cv2
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Callable
import re
import numpy as np



try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception:
    tk = None


@dataclass
class Op:
    name: str
    raw: str

@dataclass
class Step:
    ops: List[Op]
    @property
    def suffix(self) -> str:
        return "+".join(op.name for op in self.ops)


def pick_config_file() -> Path:
    if tk is None:
        raise RuntimeError("Tkinter not founds. Use: pip install tk")
    root = tk.Tk(); root.withdraw()
    p = filedialog.askopenfilename(title="Pick config file",
        filetypes=[("Text files","*.txt *.cfg *.conf *.ini *.config"),("All files","*.*")])
    root.destroy()
    if not p: raise FileNotFoundError("Config file not selected.")
    return Path(p)

def pick_input_folder() -> Path:
    if tk is None:
        raise RuntimeError("Tkinter not founds. Use: pip install tk")
    root = tk.Tk(); root.withdraw()
    p = filedialog.askdirectory(title="Pick IMG Directory")
    root.destroy()
    if not p: raise NotADirectoryError("Input directory not selected.")
    return Path(p)

def read_config_lines(cfg: Path) -> List[str]:
    with cfg.open("r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

def parse_steps(lines: List[str]) -> List[Step]:
    steps: List[Step] = []
    for idx, ln in enumerate(lines, start=1):
        parts = [p.strip() for p in ln.split("|")]
        ops: List[Op] = []
        for p in parts:
            tok = p.split(maxsplit=1)
            name = tok[0].strip().capitalize()
            raw = tok[1].strip() if len(tok) > 1 else ""
            ops.append(Op(name, raw))
        steps.append(Step(ops))
    if not steps:
        raise ValueError("No operations in config.")
    return steps

def ensure_output_dir(inp: Path) -> Path:
    out = inp.parent / f"{inp.name}_aug"
    out.mkdir(exist_ok=True)
    return out

def list_jpgs(inp: Path):
    return sorted([Path(p) for p in glob.glob(str(inp / "*")) if p.lower().endswith(".jpg")])

def op_dummy(img, raw): return img.copy()

def op_rotation(img, raw):
    ang = float(raw)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), ang, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def op_resize(img, raw):
    m = re.fullmatch(r"\s*(\d+)\s*[xX ]\s*(\d+)\s*", raw or "")
    if not m: raise ValueError("expected WxH (ex: 320x240)")
    w, h = int(m.group(1)), int(m.group(2))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

def op_brightness(img, raw):
    beta = float(raw) if raw else 0.0
    out = img.astype(np.int16) + int(beta)
    np.clip(out, 0, 255, out)
    return out.astype(np.uint8)

def op_contrast(img, raw):
    alpha = float(raw) if raw else 1.0
    out = (alpha * (img.astype(np.float32) - 128.0) + 128.0)
    return np.clip(out, 0, 255).astype(np.uint8)

def op_grayscale(img, raw):
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    y = 0.114 * b + 0.587 * g + 0.299 * r
    y = np.clip(y, 0, 255).astype(np.uint8)
    return np.dstack([y, y, y])

def op_fliph(img, raw):
    return img[:, ::-1, :]


OPS: Dict[str, Callable] = {

    "dummy":      op_dummy,

    "brightness": op_brightness,
    "contrast":   op_contrast,
    "grayscale":  op_grayscale,

    "fliph":      op_fliph,
    "rotation": op_rotation,
    "resize": op_resize,
}


def apply_op(img, op: Op):
    fn = OPS.get(op.name.lower())
    if fn is None:
        raise ValueError(f"Unknown operation '{op.name}'. Available: {', '.join(sorted(OPS.keys()))}")
    return fn(img, op.raw)

def run_step(img, step: Step):
    out = img
    for op in step.ops:
        out = apply_op(out, op)
    return out

def main():
    try:
        cfg = pick_config_file()
        inp = pick_input_folder()
        steps = parse_steps(read_config_lines(cfg))
        imgs = list_jpgs(inp)
        if not imgs:
            raise ValueError(f".jpg files not found in {inp}")
        out_dir = ensure_output_dir(inp)

        counter = 1
        for step in steps:
            for p in imgs:
                im = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if im is None:
                    print(f"[WARN] can not read {p.name}, jump.")
                    continue
                try:
                    aug = run_step(im, step)
                except Exception as e:
                    print(f"[ERROR] '{step.suffix}' pe {p.name}: {e}")
                    continue
                name = f"{p.stem}_{step.suffix}_{counter}.jpg"
                if cv2.imwrite(str(out_dir / name), aug, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    print(f"[OK] {name}")
                    counter += 1
                else:
                    print(f"[ERROR] Saving failed: {name}")

        if tk:
            root = tk.Tk(); root.withdraw()
            messagebox.showinfo("DONE", f"Saved {counter-1} imgs in:\n{out_dir}")
            root.destroy()
        else:
            print(f"Saved {counter-1} imgs in {out_dir}")

    except Exception as e:
        if tk:
            root = tk.Tk(); root.withdraw()
            messagebox.showerror("ERR", str(e))
            root.destroy()
        else:
            print("ERR:", e)

if __name__ == "__main__":
    main()
