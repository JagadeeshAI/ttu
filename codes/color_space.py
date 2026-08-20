"""Problem 1: Color Space - RGB to YCbCr, per-channel variance, chroma sub-sampling.

Usage:
    python3 color_space.py path/to/fabric.png
"""
import sys
import numpy as np
from PIL import Image


def rgb_to_ycbcr(img_rgb: np.ndarray) -> np.ndarray:
    """Convert an 8-bit RGB image (H,W,3) to YCbCr using the ITU-R BT.601 full-range
    convention (same as PIL's 'YCbCr' mode)."""
    return np.array(Image.fromarray(img_rgb, mode="RGB").convert("YCbCr"))


def channel_variance(img_ycbcr: np.ndarray) -> dict:
    y, cb, cr = img_ycbcr[..., 0], img_ycbcr[..., 1], img_ycbcr[..., 2]
    return {
        "Y": float(np.var(y.astype(np.float64))),
        "Cb": float(np.var(cb.astype(np.float64))),
        "Cr": float(np.var(cr.astype(np.float64))),
    }


def subsample_channel(channel: np.ndarray) -> np.ndarray:
    """Replace every 2x2 block with its average, keeping the original resolution."""
    h, w = channel.shape
    h2, w2 = h - (h % 2), w - (w % 2)
    c = channel[:h2, :w2].astype(np.float64)

    blocks = c.reshape(h2 // 2, 2, w2 // 2, 2)
    block_avg = blocks.mean(axis=(1, 3))

    out = np.repeat(np.repeat(block_avg, 2, axis=0), 2, axis=1)

    result = channel.astype(np.float64).copy()
    result[:h2, :w2] = out
    return np.clip(np.round(result), 0, 255).astype(np.uint8)


def main(path: str) -> None:
    img_rgb = np.array(Image.open(path).convert("RGB"))
    print(f"Loaded {path}: shape={img_rgb.shape}, dtype={img_rgb.dtype}")

    # (a) Convert to YCbCr
    img_ycbcr = rgb_to_ycbcr(img_rgb)
    Image.fromarray(img_ycbcr, mode="YCbCr").convert("RGB").save("fabric_ycbcr_asRGB.png")
    np.save("fabric_ycbcr.npy", img_ycbcr)

    # (b) Per-channel variance
    variances = channel_variance(img_ycbcr)
    print("\n(b) Per-channel variance (YCbCr):")
    for k, v in variances.items():
        print(f"  {k}: {v:.3f}")

    # (c) Chroma sub-sampling on Cb, Cr (2x2 block averaging)
    y, cb, cr = img_ycbcr[..., 0], img_ycbcr[..., 1], img_ycbcr[..., 2]
    cb_sub = subsample_channel(cb)
    cr_sub = subsample_channel(cr)
    img_ycbcr_sub = np.stack([y, cb_sub, cr_sub], axis=-1)

    img_rgb_sub = np.array(Image.fromarray(img_ycbcr_sub, mode="YCbCr").convert("RGB"))
    Image.fromarray(img_rgb_sub).save("fabric_chroma_subsampled.png")
    print("\n(c) Wrote fabric_chroma_subsampled.png (Y kept full-res, Cb/Cr 2x2-block-averaged)")

    diff = img_rgb.astype(np.float64) - img_rgb_sub.astype(np.float64)
    print(f"    Mean abs RGB difference vs original: {np.mean(np.abs(diff)):.4f}")
    print(f"    Max abs RGB difference vs original:  {np.max(np.abs(diff)):.1f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 color_space.py path/to/fabric.png")
        sys.exit(1)
    main(sys.argv[1])
