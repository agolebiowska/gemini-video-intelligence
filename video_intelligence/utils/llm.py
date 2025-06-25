import json
import time
from collections import deque
from dataclasses import dataclass

from google import genai
from google.genai import types
from video_intelligence.processing.types import Detection


@dataclass
class LLMResponse:
    content: list[dict]
    elapsed_time: float


class LLMCaller:
    def __init__(
        self,
        model: str,
        project_id: str,
    ) -> None:
        self._client = genai.Client(
            vertexai=True, project=project_id, location="global"
        )
        self._model = model
        self._response_mime_type = "application/json"
        self._max_output_tokens = 65535
        self._temperature = 0.0
        self._top_p = 1.0
        self._last_call_timestamp: float = 0
        self._rps_limit = 100.0
        self._rpm_limit = 1000
        self._request_timestamps = deque(maxlen=self._rpm_limit)

    def call_llm(
        self,
        data: bytes,
        prompt: str,
        response_schema: dict[str, dict] | None = None,
        mime_type: str = "image/png",
    ) -> LLMResponse | None:
        self._rate_limit()

        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(mime_type=mime_type, data=data),
                        types.Part.from_text(text=prompt),
                    ],
                )
            ]

            generation_config = types.GenerateContentConfig(
                temperature=self._temperature,
                max_output_tokens=self._max_output_tokens,
                response_mime_type=self._response_mime_type,
            )
            if response_schema:
                generation_config = types.GenerateContentConfig(
                    temperature=self._temperature,
                    max_output_tokens=self._max_output_tokens,
                    response_mime_type=self._response_mime_type,
                    response_schema=response_schema,
                )

            start_time = time.time()

            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=generation_config,
            )
            content = json.loads(response.text)

            end_time = time.time()
            elapsed_time = end_time - start_time
            self._last_call_timestamp = end_time

            return LLMResponse(content=content, elapsed_time=elapsed_time)

        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response: {e}")
        except Exception as e:
            print(f"Error processing: {e}")

    def _rate_limit(self):
        current_time = time.time()
        time_since_last_call = current_time - self._last_call_timestamp
        if time_since_last_call < 1.0 / self._rps_limit:
            sleep_time = (1.0 / self._rps_limit) - time_since_last_call
            print(f"Per-second rate limiter waiting: {sleep_time:.2f}s...")
            time.sleep(sleep_time)

        while (
            self._request_timestamps
            and current_time - self._request_timestamps[0] > 60
        ):
            self._request_timestamps.popleft()

        if len(self._request_timestamps) >= self._rpm_limit:
            wait_time = 60 - (current_time - self._request_timestamps[0])
            if wait_time > 0:
                print(f"Per-minute rate limiter waiting: {wait_time:.2f}s...")
                time.sleep(wait_time)
                current_time = time.time()

        self._request_timestamps.append(current_time)
        self._last_call_timestamp = current_time

    @staticmethod
    def parse_response(response: LLMResponse) -> list[Detection]:
        detections = []
        for item in response.content:
            detections.append(
                Detection(
                    box_2d=item["box_2d"],
                    label=item["label"],
                    confidence=item["confidence"],
                )
            )
        return detections
