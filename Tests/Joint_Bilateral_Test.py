import cv2 as cv
import time
from pathlib import Path
import numpy as np


def save_image(path: Path, img: np.ndarray, desc: str) -> None:
    ok = cv.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"Failed to save {desc}: {path}")
    print(f"[saved] {desc}: {path}")


def stage_time_ms(t0: float, t1: float) -> float:
    return (t1 - t0) * 1000.0


def main():
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir / "jbf_pipeline_outputs"
    out_dir.mkdir(exist_ok=True)

    guide_path = script_dir / "grayscale.png"
    src_path = script_dir / "LR_pseudo_heatmap.png"

    print("guide path:", guide_path)
    print("src path  :", src_path)

    if not guide_path.exists():
        raise FileNotFoundError(f"Guide image not found: {guide_path}")
    if not src_path.exists():
        raise FileNotFoundError(f"Source image not found: {src_path}")

    if not hasattr(cv, "ximgproc") or not hasattr(cv.ximgproc, "jointBilateralFilter"):
        raise RuntimeError("cv.ximgproc.jointBilateralFilter is not available.")

    total_t0 = time.perf_counter()

    t0 = time.perf_counter()
    guide_bgr = cv.imread(str(guide_path), cv.IMREAD_COLOR)
    thermal_bgr = cv.imread(str(src_path), cv.IMREAD_COLOR)
    t1 = time.perf_counter()

    if guide_bgr is None:
        raise RuntimeError(f"Failed to read guide image: {guide_path}")
    if thermal_bgr is None:
        raise RuntimeError(f"Failed to read source image: {src_path}")

    read_ms = stage_time_ms(t0, t1)

    print("guide_bgr shape  :", guide_bgr.shape)
    print("thermal_bgr shape:", thermal_bgr.shape)

    save_image(out_dir / "01_guide_bgr.png", guide_bgr, "guide_bgr")
    save_image(out_dir / "02_thermal_bgr.png", thermal_bgr, "thermal_bgr")

    t0 = time.perf_counter()
    guide_u8 = cv.cvtColor(guide_bgr, cv.COLOR_BGR2GRAY)
    t1 = time.perf_counter()
    guide_gray_ms = stage_time_ms(t0, t1)

    save_image(out_dir / "03_guide_u8_gray.png", guide_u8, "guide_u8_gray")

    t0 = time.perf_counter()
    thermal_gray = cv.cvtColor(thermal_bgr, cv.COLOR_BGR2GRAY)
    t1 = time.perf_counter()
    thermal_gray_ms = stage_time_ms(t0, t1)

    save_image(out_dir / "04_thermal_gray_from_pseudocolor.png", thermal_gray, "thermal_gray_from_pseudocolor")

    p_low = 2.0
    p_high = 98.0

    t0 = time.perf_counter()
    lo = np.percentile(thermal_gray, p_low)
    hi = np.percentile(thermal_gray, p_high)
    if hi <= lo + 1e-6:
        hi = lo + 1.0

    thermal_norm = np.clip(
        (thermal_gray.astype(np.float32) - lo) / (hi - lo),
        0.0,
        1.0
    )
    th8_low = (thermal_norm * 255.0).astype(np.uint8)
    t1 = time.perf_counter()
    normalize_ms = stage_time_ms(t0, t1)

    print(f"percentile normalize: p_low={p_low}, p_high={p_high}, lo={lo:.3f}, hi={hi:.3f}")

    save_image(out_dir / "05_th8_low_normalized.png", th8_low, "th8_low_normalized")

    t0 = time.perf_counter()
    th8_up = cv.resize(
        th8_low,
        (guide_u8.shape[1], guide_u8.shape[0]),
        interpolation=cv.INTER_LINEAR
    )
    t1 = time.perf_counter()
    upsample_ms = stage_time_ms(t0, t1)

    save_image(out_dir / "06_th8_up_interlinear.png", th8_up, "th8_up_interlinear")

    jb_d = 5
    jb_sigma_color = 25.0
    jb_sigma_space = 7.0

    t0 = time.perf_counter()
    th8_ref = cv.ximgproc.jointBilateralFilter(
        guide_u8,
        th8_up,
        jb_d,
        jb_sigma_color,
        jb_sigma_space,
    )
    t1 = time.perf_counter()
    jbf_ms = stage_time_ms(t0, t1)

    th8_ref = np.clip(th8_ref, 0, 255).astype(np.uint8)

    save_image(out_dir / "07_th8_ref_jbf.png", th8_ref, "th8_ref_jbf")

    t0 = time.perf_counter()
    diff = cv.absdiff(th8_up, th8_ref)
    diff_vis = cv.normalize(diff, None, 0, 255, cv.NORM_MINMAX)
    t1 = time.perf_counter()
    diff_ms = stage_time_ms(t0, t1)

    save_image(out_dir / "08_diff_raw.png", diff, "diff_raw")
    save_image(out_dir / "09_diff_vis.png", diff_vis, "diff_vis")

    t0 = time.perf_counter()
    overlay_noedge = cv.applyColorMap(th8_up, cv.COLORMAP_INFERNO)
    overlay_jbf = cv.applyColorMap(th8_ref, cv.COLORMAP_INFERNO)
    t1 = time.perf_counter()
    colormap_ms = stage_time_ms(t0, t1)

    save_image(out_dir / "10_overlay_noedge_inferno.png", overlay_noedge, "overlay_noedge_inferno")
    save_image(out_dir / "11_overlay_jbf_inferno.png", overlay_jbf, "overlay_jbf_inferno")

    alpha = 0.35

    t0 = time.perf_counter()
    fused_noedge = cv.addWeighted(guide_bgr, 1.0 - alpha, overlay_noedge, alpha, 0.0)
    fused_jbf = cv.addWeighted(guide_bgr, 1.0 - alpha, overlay_jbf, alpha, 0.0)
    t1 = time.perf_counter()
    fuse_ms = stage_time_ms(t0, t1)

    save_image(out_dir / "12_fused_noedge.png", fused_noedge, "fused_noedge")
    save_image(out_dir / "13_fused_jbf.png", fused_jbf, "fused_jbf")

    total_t1 = time.perf_counter()
    total_ms = stage_time_ms(total_t0, total_t1)

    print("\n================ TIMING REPORT ================")
    print(f"read images                : {read_ms:.3f} ms")
    print(f"guide -> gray              : {guide_gray_ms:.3f} ms")
    print(f"thermal -> gray            : {thermal_gray_ms:.3f} ms")
    print(f"percentile normalize       : {normalize_ms:.3f} ms")
    print(f"resize to guide size       : {upsample_ms:.3f} ms")
    print(f"jointBilateralFilter only  : {jbf_ms:.3f} ms")
    print(f"diff generation            : {diff_ms:.3f} ms")
    print(f"applyColorMap              : {colormap_ms:.3f} ms")
    print(f"fusion addWeighted         : {fuse_ms:.3f} ms")
    print(f"TOTAL pipeline time        : {total_ms:.3f} ms")
    print("==============================================\n")

    print("Output folder:", out_dir)


if __name__ == "__main__":
    main()