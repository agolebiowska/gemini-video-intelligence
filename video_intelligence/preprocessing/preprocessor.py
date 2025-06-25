import os
from dataclasses import dataclass
import cv2

from video_intelligence.utils import io, ocv, ffmpeg
from video_intelligence.processing.types import Crop


@dataclass
class VideoConfig:
    target_width: int | None = None
    target_height: int | None = None
    target_fps: float | None = None
    crop: Crop | None = None
    max_frames: int | None = None
    chunk_duration: int | None = None


class VideoPreprocessor:
    def __init__(
        self,
        source_io: io.FileSystemIO,
        target_io: io.FileSystemIO,
        config: VideoConfig,
    ) -> None:
        self._source_io = source_io
        self._target_io = target_io
        self._config = config
        self._local_io = io.LocalIO(root_path="/tmp/videos")

    def _chunk_video(self, video_path: str, video_name: str) -> list[str]:
        """Split the video into chunks of specified duration using ffmpeg."""
        output_dir = self._local_io._root_path
        output_template = os.path.join(
            output_dir, f"{video_name}_chunk_%02d.mp4"
        )

        duration = ffmpeg.get_video_duration(video_path)
        print(f"Video duration: {duration} seconds.")

        if self._config.chunk_duration:
            total_chunks = int(duration // self._config.chunk_duration)
            last_chunk_duration = duration % self._config.chunk_duration
            if last_chunk_duration > 0:
                total_chunks += 1

            ffmpeg.split_video(
                path=video_path,
                output_template=output_template,
                duration=self._config.chunk_duration,
            )
        else:
            total_chunks = 0

        chunk_files = sorted(
            [
                os.path.join(output_dir, f)
                for f in os.listdir(output_dir)
                if f.startswith(video_name) and "chunk" in f
            ]
        )

        if len(chunk_files) > total_chunks:
            print(f"Unexpected files detected. Cleaning up extra chunks.")
            chunk_files = chunk_files[:total_chunks]

        print(f"Video split into {len(chunk_files)} chunks.")
        return chunk_files

    def preprocess(
        self,
        source_path: str,
        target_path: str,
        start_frame: int | None = None,
    ) -> None:
        """Preprocess video with resizing, FPS change, cropping, and optional chunking.

        Args:
            source_path: Path to source video in source filesystem
            target_path: Base path for saving processed video(s)
            start_frame: Optional frame number to start processing from
        """
        try:
            video_content = self._source_io.get_video(source_path)
            temp_input = os.path.join(
                self._local_io._root_path, "temp_input.mp4"
            )
            if isinstance(video_content, str):
                with open(video_content, "rb") as f:
                    video_content = f.read()
            self._local_io.save_video(video_content, "temp_input.mp4")

            video_name = os.path.splitext(os.path.basename(source_path))[0]
            destination = f"{target_path}/{video_name}"

            cap = ocv.init_video_capture(temp_input)
            source_fps, width, height, _, rotation = (
                ocv.get_video_properties(cap)
            )

            if self._config.target_width and self._config.target_height:
                width = self._config.target_width
                height = self._config.target_height

            target_fps = (
                self._config.target_fps
                if self._config.target_fps
                else source_fps
            )
            frame_interval = source_fps / target_fps if target_fps else 1

            if self._config.crop:
                width = self._config.crop.width
                height = self._config.crop.height

            print(
                f"Output properties: FPS={target_fps}, Width={width}, Height={height}"
            )

            output_path = os.path.join(
                self._local_io._root_path, f"{video_name}.mp4"
            )
            out, temp_output = ocv.init_video_writer(
                output_path, target_fps, width, height
            )

            current_frame = start_frame if start_frame is not None else 0
            if start_frame is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            next_frame_to_process = current_frame
            frames_written = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or (
                    self._config.max_frames
                    and frames_written >= self._config.max_frames
                ):
                    break

                if current_frame >= next_frame_to_process:
                    if rotation:
                        if rotation == 90:
                            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                        elif rotation == 180:
                            frame = cv2.rotate(frame, cv2.ROTATE_180)
                        elif rotation == 270:
                            frame = cv2.rotate(
                                frame, cv2.ROTATE_90_COUNTERCLOCKWISE
                            )
                    if (
                        self._config.target_width
                        and self._config.target_height
                    ):
                        frame = cv2.resize(
                            frame,
                            (
                                self._config.target_width,
                                self._config.target_height,
                            ),
                        )

                    if self._config.crop:
                        h, w, _ = frame.shape
                        frame = frame[
                            self._config.crop.top : h
                            - self._config.crop.bottom,
                            self._config.crop.left : w
                            - self._config.crop.right,
                        ]

                    out.write(frame)
                    frames_written += 1
                    next_frame_to_process = current_frame + frame_interval

                current_frame += 1

            print(f"Processed {frames_written} frames")
            out.release()

            ffmpeg.encode_output(temp_output, output_path)
            print(f"Encoded to: {output_path}")
            with open(output_path, "rb") as f:
                content = f.read()

            self._target_io.save_video(
                content, f"{destination}/{video_name}.mp4"
            )
            print(f"Saved processed video to: {destination}")

            cap.release()
            cv2.destroyAllWindows()

            if self._config.chunk_duration:
                print("Chunking...")
                chunk_paths = self._chunk_video(output_path, video_name)

                for chunk_path in chunk_paths:
                    chunk_name = os.path.splitext(
                        os.path.basename(chunk_path)
                    )[0]
                    with open(chunk_path, "rb") as f:
                        content = f.read()

                    self._target_io.save_video(
                        content, f"{destination}/chunks/{chunk_name}.mp4"
                    )
                    print(f"Saved video chunk to: {destination}/chunks")
                    os.remove(chunk_path)

            os.remove(temp_output)
            os.remove(temp_input)
            os.remove(output_path)

        except Exception as e:
            print(f"Error preprocessing video: {e}")
            raise
