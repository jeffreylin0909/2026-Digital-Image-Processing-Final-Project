import argparse
from pathlib import Path

import cv2
import numpy as np


def normalize_map(arr: np.ndarray, percentile=95):
    scale = np.percentile(arr, percentile) + 1e-6
    return np.clip(arr / scale, 0, 1).astype(np.float32)


def inverse_confidence(arr: np.ndarray, percentile=95):
    return 1.0 - normalize_map(arr, percentile=percentile)


def get_single_frame_fog_components(
    image_bgr,
    patch_size=15,
    w_dcp=0.5,
    w_grad=0.3,
    w_sat=0.2,
    blur_ksize=101,
    threshold=0.6,
    morph_ksize=31,
):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0

    min_channel = np.min(image_bgr.astype(np.float32) / 255.0, axis=2)
    kernel_dcp = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    conf_dcp = cv2.erode(min_channel, kernel_dcp)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    grad_norm = np.clip(grad_mag / (np.percentile(grad_mag, 95) + 1e-6), 0, 1).astype(
        np.float32
    )
    conf_grad = 1.0 - grad_norm

    sat = hsv[:, :, 1]
    conf_sat = 1.0 - sat

    total_w = w_dcp + w_grad + w_sat
    combined = (
        (w_dcp / total_w) * conf_dcp
        + (w_grad / total_w) * conf_grad
        + (w_sat / total_w) * conf_sat
    )

    val = hsv[:, :, 2]
    val_gate = np.clip(val * 2.0, 0, 1)
    confidence_pre_blur = combined * val_gate
    confidence_map = cv2.GaussianBlur(
        confidence_pre_blur, (blur_ksize, blur_ksize), 0
    )

    binary_mask = (confidence_map > threshold).astype(np.uint8) * 255
    morph_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize)
    )
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, morph_kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, morph_kernel)

    return {
        "min_channel": min_channel,
        "conf_dcp": conf_dcp,
        "grad_norm": grad_norm,
        "conf_grad": conf_grad,
        "sat": sat,
        "conf_sat": conf_sat,
        "val": val,
        "val_gate": val_gate,
        "combined": combined,
        "confidence_pre_blur": confidence_pre_blur,
        "confidence_map": confidence_map,
        "binary_mask": binary_mask,
    }


def estimate_ecc_warp(template_gray: np.ndarray, input_gray: np.ndarray, max_dim=480):
    h, w = template_gray.shape
    scale = 1.0
    max_hw = max(h, w)
    if max_hw > max_dim:
        scale = max_dim / max_hw

    if scale != 1.0:
        size = (int(w * scale), int(h * scale))
        template_small = cv2.resize(template_gray, size, interpolation=cv2.INTER_AREA)
        input_small = cv2.resize(input_gray, size, interpolation=cv2.INTER_AREA)
    else:
        template_small = template_gray
        input_small = input_gray

    template_small = cv2.GaussianBlur(template_small, (5, 5), 0)
    input_small = cv2.GaussianBlur(input_small, (5, 5), 0)

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        60,
        1e-4,
    )

    try:
        cv2.findTransformECC(
            template_small.astype(np.float32),
            input_small.astype(np.float32),
            warp,
            cv2.MOTION_AFFINE,
            criteria,
        )
    except cv2.error:
        return np.eye(2, 3, dtype=np.float32)

    if scale != 1.0:
        warp[0, 2] /= scale
        warp[1, 2] /= scale
    return warp


def warp_to_template(img: np.ndarray, warp: np.ndarray, out_size):
    out_w, out_h = out_size
    return cv2.warpAffine(
        img,
        warp,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )


def compute_temporal_persistence_components(
    video_path: Path,
    frame_idx: int,
    center_frame_bgr: np.ndarray,
    center_comps: dict,
    *,
    patch_size,
    w_dcp,
    w_grad,
    w_sat,
    blur_ksize,
    threshold,
    morph_ksize,
    temporal_radius,
    temporal_stride,
    temporal_center_weight,
    temporal_mean_weight,
    temporal_persistence_weight,
    temporal_threshold,
    support_threshold_scale,
    support_threshold_floor,
    temporal_support_gate,
    temporal_stability_gate,
    appearance_stability_weight,
    appearance_stability_gate,
):
    h, w = center_frame_bgr.shape[:2]
    center_gray = cv2.cvtColor(center_frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    center_score = center_comps["confidence_map"].astype(np.float32)

    stack = [center_score]
    gray_stack = [center_gray]
    used_indices = [frame_idx]

    for offset in range(-temporal_radius, temporal_radius + 1):
        if offset == 0:
            continue
        neighbor_idx = frame_idx + offset * temporal_stride
        if neighbor_idx < 0:
            continue

        try:
            neighbor_frame, total_frames = read_frame(video_path, neighbor_idx)
        except RuntimeError:
            continue

        if neighbor_idx >= total_frames:
            continue

        neighbor_gray = (
            cv2.cvtColor(neighbor_frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        )
        neighbor_comps = get_single_frame_fog_components(
            neighbor_frame,
            patch_size=patch_size,
            w_dcp=w_dcp,
            w_grad=w_grad,
            w_sat=w_sat,
            blur_ksize=blur_ksize,
            threshold=threshold,
            morph_ksize=morph_ksize,
        )
        warp = estimate_ecc_warp(center_gray, neighbor_gray)
        aligned_score = warp_to_template(
            neighbor_comps["confidence_map"].astype(np.float32),
            warp,
            (w, h),
        )
        aligned_gray = warp_to_template(
            neighbor_gray.astype(np.float32),
            warp,
            (w, h),
        )
        stack.append(aligned_score)
        gray_stack.append(aligned_gray)
        used_indices.append(neighbor_idx)

    score_stack = np.stack(stack, axis=0)
    gray_stack = np.stack(gray_stack, axis=0)
    temporal_mean = np.mean(score_stack, axis=0)
    temporal_std = np.std(score_stack, axis=0)
    temporal_stability = inverse_confidence(temporal_std, percentile=95)
    appearance_std = np.std(gray_stack, axis=0)
    appearance_stability = inverse_confidence(appearance_std, percentile=95)

    support_thr = max(support_threshold_floor, threshold * support_threshold_scale)
    temporal_support = np.mean(score_stack > support_thr, axis=0).astype(np.float32)
    temporal_persistence = (
        temporal_mean
        * temporal_stability
        * temporal_support
        * (
            (1.0 - appearance_stability_weight)
            + appearance_stability_weight * appearance_stability
        )
    )

    total_w = (
        temporal_center_weight
        + temporal_mean_weight
        + temporal_persistence_weight
    )
    temporal_score = (
        (temporal_center_weight / total_w) * center_score
        + (temporal_mean_weight / total_w) * temporal_mean
        + (temporal_persistence_weight / total_w) * temporal_persistence
    )
    temporal_score = (
        cv2.bilateralFilter(
            np.clip(temporal_score * 255.0, 0, 255).astype(np.uint8),
            d=9,
            sigmaColor=25,
            sigmaSpace=21,
        ).astype(np.float32)
        / 255.0
    )

    temporal_mask = (
        (temporal_score > temporal_threshold)
        & (temporal_support > temporal_support_gate)
        & (temporal_stability > temporal_stability_gate)
        & (appearance_stability > appearance_stability_gate)
    ).astype(np.uint8) * 255
    morph_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize)
    )
    temporal_mask = cv2.morphologyEx(temporal_mask, cv2.MORPH_CLOSE, morph_kernel)
    temporal_mask = cv2.morphologyEx(temporal_mask, cv2.MORPH_OPEN, morph_kernel)

    return {
        "temporal_mean": temporal_mean,
        "temporal_std": temporal_std,
        "temporal_stability": temporal_stability,
        "appearance_std": appearance_std,
        "appearance_stability": appearance_stability,
        "temporal_support": temporal_support,
        "temporal_persistence": temporal_persistence,
        "temporal_score": temporal_score,
        "temporal_mask": temporal_mask,
        "temporal_indices": used_indices,
    }


def read_frame(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx < 0:
        frame_idx = max(total_frames + frame_idx, 0)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
    return frame, total_frames


def to_heatmap(img_float: np.ndarray):
    img_u8 = np.clip(img_float * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(img_u8, cv2.COLORMAP_TURBO)


def to_gray_bgr(img_float: np.ndarray):
    img_u8 = np.clip(img_float * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)


def make_overlay(image_bgr: np.ndarray, binary_mask: np.ndarray):
    overlay = image_bgr.copy()
    red = np.zeros_like(overlay)
    red[:, :, 2] = 255
    mask_bool = binary_mask > 0
    overlay[mask_bool] = cv2.addWeighted(overlay, 0.45, red, 0.55, 0)[mask_bool]
    return overlay


def draw_panel(title: str, image_bgr: np.ndarray, panel_size):
    panel_w, panel_h = panel_size
    resized = cv2.resize(image_bgr, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(resized, (0, 0), (panel_w - 1, panel_h - 1), (40, 40, 40), 1)
    cv2.rectangle(resized, (0, 0), (panel_w, 36), (18, 18, 18), -1)
    cv2.putText(
        resized,
        title,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    return resized


def build_grid(panels, cols=3, panel_size=(480, 280), gap=12, bg_color=(245, 245, 245)):
    rows = int(np.ceil(len(panels) / cols))
    panel_w, panel_h = panel_size
    canvas_w = cols * panel_w + (cols + 1) * gap
    canvas_h = rows * panel_h + (rows + 1) * gap
    canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)

    for i, (title, image_bgr) in enumerate(panels):
        row = i // cols
        col = i % cols
        x = gap + col * (panel_w + gap)
        y = gap + row * (panel_h + gap)
        canvas[y : y + panel_h, x : x + panel_w] = draw_panel(
            title, image_bgr, panel_size
        )
    return canvas


def save_stats(stats_path: Path, args, total_frames: int, comps, temporal_comps):
    confidence = comps["confidence_map"]
    mask = comps["binary_mask"] > 0
    fog_ratio = float(mask.mean())
    quantiles = np.quantile(confidence, [0.5, 0.75, 0.9, 0.95, 0.99])
    temporal_conf = temporal_comps["temporal_score"]
    temporal_mask = temporal_comps["temporal_mask"] > 0
    temporal_fog_ratio = float(temporal_mask.mean())
    temporal_quantiles = np.quantile(temporal_conf, [0.5, 0.75, 0.9, 0.95, 0.99])

    lines = [
        f"video={args.video}",
        f"frame={args.frame}",
        f"total_frames={total_frames}",
        f"patch_size={args.patch_size}",
        f"weights=dcp:{args.w_dcp},grad:{args.w_grad},sat:{args.w_sat}",
        f"blur_ksize={args.blur_ksize}",
        f"threshold={args.threshold}",
        f"morph_ksize={args.morph_ksize}",
        f"temporal_radius={args.temporal_radius}",
        f"temporal_stride={args.temporal_stride}",
        f"temporal_center_weight={args.temporal_center_weight}",
        f"temporal_mean_weight={args.temporal_mean_weight}",
        f"temporal_persistence_weight={args.temporal_persistence_weight}",
        f"temporal_threshold={args.temporal_threshold}",
        f"support_threshold_scale={args.support_threshold_scale}",
        f"support_threshold_floor={args.support_threshold_floor}",
        f"temporal_support_gate={args.temporal_support_gate}",
        f"temporal_stability_gate={args.temporal_stability_gate}",
        f"appearance_stability_weight={args.appearance_stability_weight}",
        f"appearance_stability_gate={args.appearance_stability_gate}",
        "temporal_indices="
        + ",".join(str(v) for v in temporal_comps["temporal_indices"]),
        f"fog_ratio={fog_ratio:.4f}",
        f"temporal_fog_ratio={temporal_fog_ratio:.4f}",
        "confidence_quantiles="
        + ",".join(f"{q:.4f}" for q in quantiles),
        "temporal_confidence_quantiles="
        + ",".join(f"{q:.4f}" for q in temporal_quantiles),
    ]
    stats_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the coarse fog mask and its component confidence maps."
    )
    parser.add_argument(
        "--video",
        default="data/foggy_vid/1.mp4",
        help="Path to input video.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=138,
        help="0-based frame index. Negative values count from the end.",
    )
    parser.add_argument("--patch-size", type=int, default=15)
    parser.add_argument("--w-dcp", type=float, default=0.5)
    parser.add_argument("--w-grad", type=float, default=0.3)
    parser.add_argument("--w-sat", type=float, default=0.2)
    parser.add_argument("--blur-ksize", type=int, default=101)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--morph-ksize", type=int, default=31)
    parser.add_argument("--temporal-radius", type=int, default=2)
    parser.add_argument("--temporal-stride", type=int, default=1)
    parser.add_argument("--temporal-center-weight", type=float, default=0.35)
    parser.add_argument("--temporal-mean-weight", type=float, default=0.25)
    parser.add_argument("--temporal-persistence-weight", type=float, default=0.40)
    parser.add_argument("--temporal-threshold", type=float, default=0.60)
    parser.add_argument("--support-threshold-scale", type=float, default=0.82)
    parser.add_argument("--support-threshold-floor", type=float, default=0.38)
    parser.add_argument("--temporal-support-gate", type=float, default=0.0)
    parser.add_argument("--temporal-stability-gate", type=float, default=0.0)
    parser.add_argument("--appearance-stability-weight", type=float, default=0.35)
    parser.add_argument("--appearance-stability-gate", type=float, default=0.0)
    parser.add_argument(
        "--out-prefix",
        default="output/coarse_mask_viz_v1_f138",
        help="Output prefix without extension.",
    )
    args = parser.parse_args()

    if (
        args.blur_ksize % 2 == 0
        or args.morph_ksize % 2 == 0
    ):
        raise ValueError("blur_ksize and morph_ksize must be odd.")

    video_path = Path(args.video)
    frame_bgr, total_frames = read_frame(video_path, args.frame)
    comps = get_single_frame_fog_components(
        frame_bgr,
        patch_size=args.patch_size,
        w_dcp=args.w_dcp,
        w_grad=args.w_grad,
        w_sat=args.w_sat,
        blur_ksize=args.blur_ksize,
        threshold=args.threshold,
        morph_ksize=args.morph_ksize,
    )
    temporal_comps = compute_temporal_persistence_components(
        video_path,
        args.frame,
        frame_bgr,
        comps,
        patch_size=args.patch_size,
        w_dcp=args.w_dcp,
        w_grad=args.w_grad,
        w_sat=args.w_sat,
        blur_ksize=args.blur_ksize,
        threshold=args.threshold,
        morph_ksize=args.morph_ksize,
        temporal_radius=args.temporal_radius,
        temporal_stride=args.temporal_stride,
        temporal_center_weight=args.temporal_center_weight,
        temporal_mean_weight=args.temporal_mean_weight,
        temporal_persistence_weight=args.temporal_persistence_weight,
        temporal_threshold=args.temporal_threshold,
        support_threshold_scale=args.support_threshold_scale,
        support_threshold_floor=args.support_threshold_floor,
        temporal_support_gate=args.temporal_support_gate,
        temporal_stability_gate=args.temporal_stability_gate,
        appearance_stability_weight=args.appearance_stability_weight,
        appearance_stability_gate=args.appearance_stability_gate,
    )

    overlay = make_overlay(frame_bgr, comps["binary_mask"])
    temporal_overlay = make_overlay(frame_bgr, temporal_comps["temporal_mask"])
    mask_bgr = cv2.cvtColor(comps["binary_mask"], cv2.COLOR_GRAY2BGR)
    temporal_mask_bgr = cv2.cvtColor(temporal_comps["temporal_mask"], cv2.COLOR_GRAY2BGR)

    panels = [
        ("Original Frame", frame_bgr),
        ("Single-Frame Overlay", overlay),
        ("Single-Frame Binary Mask", mask_bgr),
        ("DCP Confidence", to_heatmap(comps["conf_dcp"])),
        ("Gradient Confidence", to_heatmap(comps["conf_grad"])),
        ("Saturation Confidence", to_heatmap(comps["conf_sat"])),
        ("Value Gate", to_heatmap(comps["val_gate"])),
        ("Combined Before Blur", to_heatmap(comps["confidence_pre_blur"])),
        ("Single-Frame Fog Score", to_heatmap(comps["confidence_map"])),
        ("Temporal Mean Score", to_heatmap(temporal_comps["temporal_mean"])),
        ("Temporal Stability", to_heatmap(temporal_comps["temporal_stability"])),
        ("Appearance Stability", to_heatmap(temporal_comps["appearance_stability"])),
        ("Temporal Support", to_heatmap(temporal_comps["temporal_support"])),
        ("Temporal Persistence", to_heatmap(temporal_comps["temporal_persistence"])),
        ("Temporal Fog Score", to_heatmap(temporal_comps["temporal_score"])),
        ("Temporal Mask Overlay", temporal_overlay),
        ("Temporal Binary Mask", temporal_mask_bgr),
    ]
    grid = build_grid(panels)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    grid_path = out_prefix.with_suffix(".jpg")
    overlay_path = out_prefix.with_name(out_prefix.name + "_overlay").with_suffix(".png")
    mask_path = out_prefix.with_name(out_prefix.name + "_mask").with_suffix(".png")
    temporal_overlay_path = out_prefix.with_name(
        out_prefix.name + "_temporal_overlay"
    ).with_suffix(".png")
    temporal_mask_path = out_prefix.with_name(
        out_prefix.name + "_temporal_mask"
    ).with_suffix(".png")
    stats_path = out_prefix.with_name(out_prefix.name + "_stats").with_suffix(".txt")

    cv2.imwrite(str(grid_path), grid)
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(mask_path), comps["binary_mask"])
    cv2.imwrite(str(temporal_overlay_path), temporal_overlay)
    cv2.imwrite(str(temporal_mask_path), temporal_comps["temporal_mask"])
    save_stats(stats_path, args, total_frames, comps, temporal_comps)

    print(f"saved_grid={grid_path.resolve()}")
    print(f"saved_overlay={overlay_path.resolve()}")
    print(f"saved_mask={mask_path.resolve()}")
    print(f"saved_temporal_overlay={temporal_overlay_path.resolve()}")
    print(f"saved_temporal_mask={temporal_mask_path.resolve()}")
    print(f"saved_stats={stats_path.resolve()}")


if __name__ == "__main__":
    main()
