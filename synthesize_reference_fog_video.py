import argparse
from pathlib import Path

import cv2
import numpy as np

from visualize_coarse_fog_mask import get_single_frame_fog_components


def estimate_airlight(frame_bgr: np.ndarray):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    bgr = frame_bgr.astype(np.float32)
    val = hsv[:, :, 2]
    sat = hsv[:, :, 1]
    candidate = (val >= np.quantile(val, 0.92)) & (sat <= np.quantile(sat, 0.45))
    if np.count_nonzero(candidate) < 64:
        candidate = val >= np.quantile(val, 0.98)
    samples = bgr[candidate]
    if samples.size == 0:
        samples = bgr.reshape(-1, 3)
    return np.mean(samples, axis=0)


def build_reference_fog_layer(frame_bgr: np.ndarray):
    comps = get_single_frame_fog_components(
        frame_bgr,
        patch_size=15,
        w_dcp=0.5,
        w_grad=0.3,
        w_sat=0.2,
        blur_ksize=101,
        threshold=0.6,
        morph_ksize=31,
    )
    conf = comps["confidence_map"].astype(np.float32)
    mask = comps["binary_mask"].astype(np.float32) / 255.0

    conf_alpha = np.clip((conf - 0.42) / 0.32, 0.0, 1.0)
    soft_mask = cv2.GaussianBlur(mask, (0, 0), 18)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    lum = cv2.GaussianBlur(gray, (0, 0), 21)
    lum = (lum - lum.min()) / (lum.max() - lum.min() + 1e-6)

    # Use the real fog confidence as the main shape, with a gentle luminance modulation
    # so the synthetic veil keeps the soft, translucent look of the captured fog.
    alpha = 0.68 * soft_mask + 0.32 * conf_alpha
    alpha = alpha * (0.88 + 0.18 * lum)
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 11)
    return alpha, estimate_airlight(frame_bgr)


def synthesize_frame(clean_bgr: np.ndarray, alpha: np.ndarray, airlight_bgr: np.ndarray):
    clean = clean_bgr.astype(np.float32)
    alpha_3 = alpha[..., None]
    fogged = clean * (1.0 - alpha_3) + airlight_bgr.reshape(1, 1, 3) * alpha_3

    # Add a mild low-frequency glow inside the fog to mimic the milky bloom.
    glow = cv2.GaussianBlur(clean_bgr, (0, 0), 9).astype(np.float32)
    fogged = fogged * (1.0 - 0.12 * alpha_3) + glow * (0.12 * alpha_3)
    return np.clip(fogged, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="Transfer the look of the real captured fog onto a clean video."
    )
    parser.add_argument("--clean-video", default="data/fogless_vid/j2.mp4")
    parser.add_argument("--reference-fog-video", default="data/foggy_vid/1.mp4")
    parser.add_argument("--out-video", default="data/foggy_vid/j2_synth_ref_fog.mp4")
    parser.add_argument("--preview", default="output/j2_synth_ref_fog_preview.jpg")
    parser.add_argument("--start-frame", type=int, default=44)
    parser.add_argument("--num-frames", type=int, default=347)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--fog-strength", type=float, default=0.82)
    parser.add_argument("--reference-speed", type=float, default=0.35)
    parser.add_argument("--alpha-ema", type=float, default=0.88)
    parser.add_argument("--airlight-ema", type=float, default=0.92)
    args = parser.parse_args()

    clean_cap = cv2.VideoCapture(args.clean_video)
    ref_cap = cv2.VideoCapture(args.reference_fog_video)
    if not clean_cap.isOpened():
        raise RuntimeError(f"Cannot open clean video: {args.clean_video}")
    if not ref_cap.isOpened():
        raise RuntimeError(f"Cannot open reference fog video: {args.reference_fog_video}")

    total_clean = int(clean_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_ref = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = min(max(args.start_frame, 0), max(total_clean - 1, 0))
    count = min(args.num_frames, total_clean - start)

    out_path = Path(args.out_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )

    preview_frames = []
    prev_alpha = None
    prev_airlight = None
    for i in range(count):
        clean_idx = start + i
        ref_idx = int(np.clip(round(i * args.reference_speed), 0, total_ref - 1))

        clean_cap.set(cv2.CAP_PROP_POS_FRAMES, clean_idx)
        ref_cap.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
        ok_clean, clean_frame = clean_cap.read()
        ok_ref, ref_frame = ref_cap.read()
        if not ok_clean or not ok_ref:
            break

        clean_frame = cv2.resize(clean_frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
        ref_frame = cv2.resize(ref_frame, (args.width, args.height), interpolation=cv2.INTER_AREA)

        alpha, airlight = build_reference_fog_layer(ref_frame)
        if prev_alpha is not None:
            alpha = args.alpha_ema * prev_alpha + (1.0 - args.alpha_ema) * alpha
        if prev_airlight is not None:
            airlight = args.airlight_ema * prev_airlight + (1.0 - args.airlight_ema) * airlight
        prev_alpha = alpha
        prev_airlight = airlight
        alpha = np.clip(alpha * args.fog_strength, 0.0, 0.95)
        fogged = synthesize_frame(clean_frame, alpha, airlight)
        writer.write(fogged)

        if i in {0, count // 3, (2 * count) // 3, count - 1}:
            vis = np.concatenate([clean_frame, fogged], axis=1)
            cv2.putText(
                vis,
                f"clean {clean_idx} | ref fog {ref_idx}",
                (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            preview_frames.append(vis)

    writer.release()
    clean_cap.release()
    ref_cap.release()

    if preview_frames:
        preview = np.concatenate(
            [cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2)) for f in preview_frames],
            axis=0,
        )
        preview_path = Path(args.preview)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview_path), preview)

    print(f"saved_video={out_path.resolve()}")
    print(f"saved_preview={Path(args.preview).resolve()}")


if __name__ == "__main__":
    main()
