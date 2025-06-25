from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple


@dataclass
class Detection:
    """
    Represents a single object detection.

    Attributes:
        box_2d (List[int]):  The 2D bounding box in [ymin, xmin, ymax, xmax] format (normalized 0-1000).
        label (str): The label of the detected object.
    """

    box_2d: List[int]
    label: str
    confidence: float

    def to_pixel_coords(
        self, frame_width: int, frame_height: int
    ) -> Tuple[int, int, int, int]:
        """Converts normalized coordinates to pixel coordinates."""
        ymin, xmin, ymax, xmax = self.box_2d
        xmin = int(xmin / 1000 * frame_width)
        ymin = int(ymin / 1000 * frame_height)
        xmax = int(xmax / 1000 * frame_width)
        ymax = int(ymax / 1000 * frame_height)
        return ymin, xmin, ymax, xmax

    def to_dict(self):
        return {
            'box_2d': self.box_2d,
            'label': self.label,
            'confidence': self.confidence,
        }


@dataclass
class Timestamp:
    """
    Represents a collection of object detections at a specific timestamp.

    Attributes:
        timestamp (str): The timestamp in MM:SS format
        objects (List[Detection]): List of object detections at this timestamp
    """

    timestamp: str
    objects: List[Detection]

    @classmethod
    def from_dict(cls, data: dict) -> 'Timestamp':
        """
        Creates a Timestamp instance from a dictionary format.

        Args:
            data (dict): Dictionary containing timestamp and objects data

        Returns:
            Timestamp: A new Timestamp instance
        """
        return cls(
            timestamp=data["timestamp"],
            objects=[
                Detection(
                    box_2d=item["box_2d"],
                    label=item["label"],
                    confidence=item["confidence"],
                )
                for item in data["objects"]
            ],
        )

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'objects': [obj.to_dict() for obj in self.objects],
        }


@dataclass
class Sequence:
    description: str
    start_time: datetime
    end_time: datetime


@dataclass
class Crop:
    width: int
    height: int
    top: int
    bottom: int
    left: int
    right: int

    @classmethod
    def from_tuple(cls, data: Tuple[int, int, int, int, int, int]) -> 'Crop':
        """
        Creates a Crop instance from a tuple format.

        Args:
            data (tuple): Tuple containing crop parameters in the order (width, height, top, bottom, left, right)

        Returns:
            Crop: A new Crop instance
        """
        return cls(
            width=data[0],
            height=data[1],
            top=data[2],
            bottom=data[3],
            left=data[4],
            right=data[5],
        )
