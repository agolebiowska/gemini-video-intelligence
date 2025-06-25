OBJECTS_TO_DETECT = [
    "car",
    "human",
    "traffic light",
    "traffic sign",
]

# HIGH_FP_OBJECTS = ["tree", "bush", "traffic lights", "traffic sign"]

MAX_ITEMS = 15

BOX_RESPONSE = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "box_2d": {
                "type": "array",
                "items": {
                    "type": "integer",
                    "description": (
                        "Bounding box coordinates in [ymin, xmin, ymax, xmax] format (normalized 0-1000)."
                    ),
                },
            },
            "label": {
                "type": "string",
                "description": "An object label from the labels list.",
            },
            "confidence": {
                "type": "number",
                "description": (
                    "Confidence score for the detection, between 0.0 and 1.0."
                ),
            },
        },
        "required": ["box_2d", "label", "confidence"],
    },
}

TIMESTAMPS_RESPONSE = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "timestamp": {
                "type": "string",
                "description": "Video timestamp in MM:SS format",
            },
            "objects": BOX_RESPONSE,
        },
    },
}

SEQUENCE_RESPONSE = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": (
                    "Detailed description of what's happening in this sequence"
                ),
            },
            "start_time": {
                "type": "string",
                "description": "Start time in MM:SS format",
            },
            "end_time": {
                "type": "string",
                "description": "End time in MM:SS format",
            },
        },
        "required": ["description", "start_time", "end_time"],
    },
}

PROMPTS = {
    "frame_2d_bounding": {
        "prompt": (
            f"""
            TASK:
You are an expert computer vision system. Your task is to perform object detection on the user-provided image.

INSTRUCTIONS:
1. Analyze the image and identify all objects belonging to the following classes: {', '.join(OBJECTS_TO_DETECT)}.
2. For each detected object, provide its class label, a precise bounding box, and your confidence score.
3. The bounding box coordinates must be in the format [ymin, xmin, ymax, xmax], representing the absolute pixel values of the top-left and bottom-right corners. The origin (0,0) is the top-left corner of the image.
4. The confidence score must be a float between 0.0 and 1.0, indicating your certainty in the detection.
5. Report no more than {MAX_ITEMS} of the most prominent and clear objects.
            """
        ),
        "response_schema": BOX_RESPONSE,
    },
    "sequence_extraction": {
        "prompt": (
            """
You are an intelligent assistant tasked with analyzing and describing videos based on object detection data. I will provide you with:

1. A video file for visual context
2. Object detection data containing:
   - Timestamps for each frame
   - List of objects detected in each frame

Your task is to:

1. Focus primarily on the detected objects in your analysis
2. Identify key sequences in the video by analyzing when different objects appear and disappear
3. For each sequence, provide:
   - A description of what's happening, emphasizing the detected objects
   - Start and end timestamps

4. Pay special attention to:
   - Humans and their activities
   - Vehicles and their movements
   - Important objects that appear on the ground
   - Significant changes in which objects are present

For each sequence include:

**Sequence 1**
description: [Description focusing on the detected objects and what they appear to be doing]
start_time: [MM:SS format]
end_time: [MM:SS format]
...

Base your sequences on meaningful changes in the objects present. When describing the video, prioritize the objects that have been detected by the system rather than background elements. Your descriptions should align with the detected objects at each timestamp.

Aim to create 3-5 meaningful sequences that effectively summarize the key moments centered around the detected objects in the video.

The objects detected in the given video are:
{detections}
"""
        ),
        "response_schema": SEQUENCE_RESPONSE,
    },
}
