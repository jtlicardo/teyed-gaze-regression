import argparse
from pathlib import Path

import cv2
import keras
import numpy as np


def label_to_pixel(
    x: float,
    y: float,
    width: int,
    height: int,
    *,
    invert_y: bool,
) -> tuple[float, float]:
    if invert_y:
        y = -y

    cx, cy = width / 2.0, height / 2.0
    px = cx + x * (width / 2.0)
    py = cy + y * (height / 2.0)
    return px, py


def prepare_frame(
    frame_bgr: np.ndarray,
    input_size: int,
) -> np.ndarray:
    resized = cv2.resize(
        frame_bgr, (input_size, input_size), interpolation=cv2.INTER_AREA
    )
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    arr = resized.astype(np.float32) / 255.0
    arr = (arr - 0.5) * 2.0
    return arr


def run_prediction(model: keras.Model, input_frame: np.ndarray) -> tuple[float, float]:
    batch = np.expand_dims(input_frame, axis=0)
    pred = model.predict(batch, verbose=0)
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    pred = np.asarray(pred).squeeze()
    if pred.size < 2:
        raise RuntimeError(f"Model output has unexpected shape: {pred.shape}")
    return float(pred[0]), float(pred[1])


def draw_prediction(
    frame: np.ndarray,
    pred_xy: tuple[float, float] | None,
    *,
    invert_y: bool,
    label: str | None,
) -> None:
    h, w = frame.shape[:2]
    center = (int(round(w / 2.0)), int(round(h / 2.0)))
    cv2.drawMarker(
        frame,
        center,
        (255, 255, 0),
        markerType=cv2.MARKER_CROSS,
        markerSize=max(12, w // 15),
        thickness=1,
    )

    if pred_xy is None:
        if label:
            cv2.putText(
                frame,
                label,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        return

    x, y = pred_xy
    px, py = label_to_pixel(
        x,
        y,
        w,
        h,
        invert_y=invert_y,
    )
    target = (int(round(px)), int(round(py)))
    cv2.arrowedLine(frame, center, target, (0, 255, 0), 2, tipLength=0.15)
    cv2.drawMarker(
        frame,
        target,
        (0, 255, 0),
        markerType=cv2.MARKER_TILTED_CROSS,
        markerSize=max(12, w // 14),
        thickness=2,
    )

    text = f"x={x:.3f} y={y:.3f}"
    if label:
        text = f"{label} | {text}"
    cv2.putText(
        frame,
        text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run gaze prediction on a video and visualize the results."
    )
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument(
        "--model",
        default="trained_models/best_overall.keras",
        help="Path to the trained Keras model.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=96,
        help="Resize frames to this square size before inference.",
    )
    parser.add_argument(
        "--invert-y",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Invert the Y sign when drawing.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the annotated video (e.g., output.mp4).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the on-screen preview window.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after processing this many frames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    print("Loading model:", model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model = keras.models.load_model(model_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    writer = None
    if args.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width == 0 or height == 0:
            raise RuntimeError("Failed to read video dimensions.")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {args.output}")

    if not args.no_display:
        cv2.namedWindow("Gaze Prediction", cv2.WINDOW_NORMAL)

    frame_index = 0
    pred_xy: tuple[float, float] | None = None
    paused = False

    try:
        while True:
            if paused:
                key = cv2.waitKey(50) & 0xFF
                if key == ord("p"):
                    paused = False
                elif key in (ord("q"), 27):
                    break
                continue

            ret, frame = cap.read()
            if not ret:
                break
            if args.max_frames is not None and frame_index >= args.max_frames:
                break

            input_frame = prepare_frame(
                frame,
                args.input_size,
            )
            pred_xy = run_prediction(model, input_frame)

            display_frame = frame.copy()
            draw_prediction(
                display_frame,
                pred_xy,
                invert_y=args.invert_y,
                label=None,
            )

            if writer is not None:
                writer.write(display_frame)

            if not args.no_display:
                cv2.imshow("Gaze prediction", display_frame)
                delay_ms = max(1, int(1000 / fps))
                key = cv2.waitKey(delay_ms) & 0xFF
                if key == ord("p"):
                    paused = True
                elif key in (ord("q"), 27):
                    break

            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(f"Processed {frame_index} frames.")


if __name__ == "__main__":
    main()
