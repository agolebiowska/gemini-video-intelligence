import os
import subprocess


def convert_to_format(
    input_path: str, output_path: str, video_codec: str = "libx264"
):
    if not os.path.exists(input_path):
        print(f"Input path does not exist: {input_path}")
        return

    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        input_path,
        "-c:v",
        video_codec,
        "-preset",
        "medium",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-y",
        output_path,
    ]
    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)


def encode_output(temp_output: str, output_path: str):
    if not os.path.exists(temp_output):
        print(f"Output path does not exist: {temp_output}")
        return

    convert_to_format(temp_output, output_path)


def get_video_duration(path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]

    return float(subprocess.check_output(cmd).decode().strip())


def split_video(path: str, output_template: str, duration: int):
    cmd = [
        "ffmpeg",
        "-i",
        path,
        "-c",
        "copy",
        "-map",
        "0",
        "-f",
        "segment",
        "-segment_time",
        str(duration),
        "-reset_timestamps",
        "1",
        output_template,
    ]
    subprocess.run(cmd, check=True)
