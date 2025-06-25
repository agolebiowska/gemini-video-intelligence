import os
import subprocess
import cv2
from tqdm import tqdm

from IPython.display import display, Image

from video_intelligence.utils import io, ocv, ffmpeg
from video_intelligence.processing.types import Timestamp
from video_intelligence.utils.llm import LLMCaller
from video_intelligence.config import Config
from video_intelligence.processing.prompts import PROMPTS


class VideoProcessor:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._llm_caller = LLMCaller(
            model=config.model_id, project_id=config.project_id
        )
        self._local_io = io.LocalIO(root_path=config.paths.tmp)

    def process_video(
        self,
        video_path: str,
        frame_by_frame: bool = True,
    ) -> tuple[str | None, list[Timestamp] | None]:
        try:
            if frame_by_frame:
                return self._process_frame_by_frame(video_path)

            return self._process_video_chunk(video_path)

        except Exception as e:
            print(f"Error in video processing: {e}")
            return None, None

    def _process_frame_by_frame(
        self,
        video_path: str,
    ) -> tuple[str | None, list[Timestamp] | None]:
        try:
            cap = ocv.init_video_capture(video_path)
            (
                fps,
                width,
                height,
                total_frames,
                _,
            ) = ocv.get_video_properties(cap)

            output_path = f"{self._config.paths.tmp}/output_with_boxes_frame_by_frame.mp4"

            out, temp_output = ocv.init_video_writer(
                output_path, fps, width, height
            )

            with tqdm(
                total=total_frames, desc="Processing Frames", unit="frame"
            ) as pbar:
                frame_count = 0
                timestamps = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        print("No more frames to read.")
                        break

                    img_data = ocv.preprocess_frame(frame)

                    try:
                        prompt_config = PROMPTS[self._config.detection_prompt]
                        response = None
                        for _ in range(3):
                            response = self._llm_caller.call_llm(
                                data=img_data,
                                prompt=prompt_config["prompt"],
                                response_schema=prompt_config[
                                    "response_schema"
                                ],
                            )
                            if response and response.content:
                                break

                        if response and response.content:
                            detections = LLMCaller.parse_response(response)
                            ocv.draw_bounding_boxes(
                                frame,
                                detections,
                                width,
                                height,
                            )

                            if frame_count % self._config.debug_interval == 0:
                                frame_with_boxes = ocv.get_frame_with_boxes(
                                    frame,
                                    detections,
                                    width,
                                    height,
                                )
                                display(Image(frame_with_boxes))

                            timestamp = Timestamp(
                                timestamp=frame_to_timestamp(frame_count, fps),
                                objects=detections,
                            )
                            timestamps.append(timestamp)

                    except Exception as e:
                        print(f"Error processing frame {frame_count + 1}: {e}")
                        continue

                    out.write(frame)

                    pbar.update(1)
                    frame_count += 1

                cap.release()
                out.release()
                cv2.destroyAllWindows()

                try:
                    if os.path.exists(temp_output):
                        ffmpeg.encode_output(temp_output, output_path)
                        os.remove(temp_output)

                except subprocess.CalledProcessError as e:
                    raise Exception(
                        f"Error during ffmpeg conversion: {e.stderr.decode()}"
                    )

                print("Processing completed. Output saved to:", output_path)
                return output_path, timestamps

        except Exception as e:
            print(f"Error processing video: {e}")
            return None, None

    def _process_video_chunk(
        self,
        video_path: str,
    ) -> tuple[str | None, list[Timestamp] | None]:
        video = self._local_io.get_video(video_path)
        prompt_config = PROMPTS[self._config.detection_prompt]
        output_path = f"{self._config.paths.tmp}/output_with_boxes.mp4"

        try:
            response = None
            for _ in range(3):
                response = self._llm_caller.call_llm(
                    data=video,
                    prompt=prompt_config["prompt"],
                    response_schema=prompt_config["response_schema"],
                    mime_type="video/mp4",
                )
                if response and response.content:
                    break
        except Exception as e:
            print(f"Error processing video chunk: {e}")
            return None, None

        if not response or not response.content:
            print("Empty response")
            return None, None

        timestamps = [Timestamp.from_dict(entry) for entry in response.content]
        print(f"Timestamps parsed: {timestamps}")

        cap = ocv.init_video_capture(video_path)
        (
            fps,
            width,
            height,
            _,
            _,
        ) = ocv.get_video_properties(cap)

        out, temp_output = ocv.init_video_writer(
            output_path, fps, width, height
        )

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count > len(timestamps):
                break

            detections = timestamps[frame_count].objects
            print(
                f"Detections for frame {frame_count}, detections: {detections}"
            )

            ocv.draw_bounding_boxes(
                frame,
                detections,
                width,
                height,
            )

            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        try:
            if os.path.exists(temp_output):
                ffmpeg.encode_output(temp_output, output_path)
                os.remove(temp_output)

        except subprocess.CalledProcessError as e:
            raise Exception(
                f"Error during ffmpeg conversion: {e.stderr.decode()}"
            )

        print("Processing completed. Output saved to:", output_path)
        return output_path, timestamps


def frame_to_timestamp(frame_count, fps) -> str:
    total_seconds = int(frame_count // fps)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"
