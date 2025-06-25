import os
import numpy as np
import cv2

from video_intelligence.processing.types import Detection


def init_video_capture(video_path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video at {video_path}")
    print("Video capture initialized.")
    return cap


def get_video_properties(
    cap: cv2.VideoCapture,
) -> tuple[float, int, int, int, int]:
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))

    if rotation == 90 or rotation == 270:
        frame_width, frame_height = frame_height, frame_width

    print(
        f"Video properties: FPS={fps}, Width={frame_width}, Height={frame_height}, Frames={total_frames}, Rotation={rotation}"
    )
    return fps, frame_width, frame_height, total_frames, rotation


def init_video_writer(
    output_path: str, fps: float, width: int, height: int, codec: str = "mp4v"
) -> tuple[cv2.VideoWriter, str]:
    filename, ext = os.path.splitext(output_path)
    ext = ext.lower()
    temp_out = filename + "_temp" + ext
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(temp_out, fourcc, fps, (width, height))

    if not out.isOpened():
        raise Exception("Failed to initialize video writer")

    return out, temp_out


def preprocess_frame(frame: np.ndarray) -> bytes:
    ret, encoded_img = cv2.imencode(".jpg", frame)
    if not ret:
        raise ValueError("Could not encode image to JPEG")
    return encoded_img.tobytes()


def draw_bounding_boxes(
    frame: np.ndarray,
    detections: list,
    width: int,
    height: int,
    inverse_colors: bool = False,
):
    unique_labels = sorted(list(set(d.label for d in detections)))
    if not unique_labels:
        return

    colors = cv2.applyColorMap(
        np.arange(0, 255, 255 / len(unique_labels)).astype(np.uint8),
        cv2.COLORMAP_RAINBOW,
    )
    color_map = {
        label: tuple(int(c) for c in colors[i][0])
        for i, label in enumerate(unique_labels)
    }

    for detection in detections:
        base_box_color = color_map.get(detection.label, (0, 255, 0))
        if inverse_colors:
            txt_color = (0, 0, 0)
            box_color = tuple(
                int(c * 0.5 + w * (1 - 0.5))
                for c, w in zip(base_box_color, (255, 255, 255))
            )
        else:
            txt_color = (255, 255, 255)
            box_color = base_box_color

        ymin, xmin, ymax, xmax = detection.to_pixel_coords(width, height)
        lw = max(round(sum(frame.shape) / 2 * 0.002), 1)

        p1, p2 = (xmin, ymin), (xmax, ymax)
        cv2.rectangle(
            frame, p1, p2, box_color, thickness=lw, lineType=cv2.LINE_AA
        )

        label = detection.label
        if label:
            tf = max(lw - 1, 1)
            font_scale_adaptive = lw / 4
            w, h = cv2.getTextSize(
                label, 0, fontScale=font_scale_adaptive, thickness=tf
            )[0]

            outside = p1[1] - h >= 3
            p2_text = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3

            cv2.rectangle(frame, p1, p2_text, box_color, -1, cv2.LINE_AA)

            text_pos = (p1[0], p1[1] - 2 if outside else p1[1] + h + 2)
            cv2.putText(
                frame,
                label,
                text_pos,
                0,
                font_scale_adaptive,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA,
            )


def get_frame_with_boxes(
    frame: np.ndarray,
    detections: list[Detection],
    width: int,
    height: int,
    inverse_colors: bool = False,
) -> bytes:
    """Displays the frame with bounding boxes (for debugging)."""
    frame_with_boxes = frame.copy()
    draw_bounding_boxes(
        frame_with_boxes, detections, width, height, inverse_colors
    )
    _, buffer = cv2.imencode(".jpg", frame_with_boxes)
    image_bytes = buffer.tobytes()
    return image_bytes
