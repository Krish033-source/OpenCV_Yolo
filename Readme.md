# Real-Time CCTV Object Detection & Event Pipeline

A video analytics prototype that detects and tracks objects in CCTV-style footage using **YOLO** and **OpenCV**, and generates structured **JSON events** when an object enters a predefined restricted zone.

<img width="613" height="502" alt="Screenshot 2026-10-07 050011" src="https://github.com/user-attachments/assets/ab7edfb1-897e-4d31-a405-92323770033b" />


## Features

- Frame-by-frame object detection with YOLO (Ultralytics)
- Confidence-based filtering (`--conf`)
- Multi-object tracking, so every object keeps a persistent `track_id`
- Bounding-box visualization with class, confidence and track ID
- Live FPS monitoring overlaid on the video
- Restricted-zone (polygon) entry detection
- Structured JSON report with video info, config, performance, summary and events
- Duplicate-event protection (warm-up frames and per-track cooldown)

## How it works

1. Read the video and process the first N seconds (`--seconds`).
2. Run YOLO with tracking on each frame and drop detections below the confidence threshold.
3. For each tracked object, take the **bottom-center point** of its bounding box (roughly where it touches the ground).
4. Check whether that point lies inside the restricted polygon.
5. If an object was **outside** in the previous frame and is **inside** now, log a `zone_entry` event.
6. Write the annotated video and the JSON report.

Two safeguards keep the events clean:

- **Warm-up:** objects already inside the zone during the first 5 frames are not counted as entries. They are reported separately as `objects_already_inside_zone_at_start`.
- **Cooldown:** the same `track_id` cannot trigger another event within `--cooldown` seconds. This stops flickering on the zone border from creating duplicates.

## Setup

```bash
pip install -r requirements.txt
```

The YOLO weights (`yolov8n.pt`) download automatically on the first run.

## Usage

```bash
python cctv_detect.py --video test.mp4 --seconds 10
```

| Argument | Default | Description |
|---|---|---|
| `--video` | required | Input video path, or `0` for webcam |
| `--model` | `yolov8n.pt` | YOLO weights file |
| `--conf` | `0.4` | Minimum confidence to keep a detection |
| `--seconds` | `10` | How many seconds of video to process |
| `--zone` | right-bottom area | Normalized polygon `x1,y1,x2,y2,...` (values between 0 and 1) |
| `--classes` | all | COCO class IDs to keep, e.g. `--classes 0 2` for person and car |
| `--cooldown` | `1.0` | Minimum seconds between two events of the same track |
| `--out` | `output.mp4` | Annotated output video |
| `--events` | `events.json` | JSON report |

Example with a custom zone (left half of the frame, persons only):

```bash
python cctv_detect.py --video test.mp4 --zone "0.05,0.3,0.5,0.3,0.5,0.95,0.05,0.95" --classes 0
```

## Output files

- `output.mp4`: video with the zone (red), bounding boxes, labels, FPS and event counter. Objects inside the zone turn red.
- `events.json`: full report, described below.

## JSON output reference

The report has five top-level sections.

### Example

> Values below are illustrative. Your numbers will differ depending on the video.

```json
{
  "video_info": {
    "source": "test.mp4",
    "resolution": {"width": 1280, "height": 720},
    "source_fps": 30.0,
    "frames_processed": 300,
    "duration_processed_sec": 10.0
  },
  "config": {
    "model": "yolov8n.pt",
    "confidence_threshold": 0.4,
    "event_cooldown_sec": 1.0,
    "warmup_frames": 5,
    "zone": {
      "name": "restricted_zone_1",
      "polygon_pixels": [[704, 252], [1216, 252], [1216, 684], [704, 684]]
    }
  },
  "performance": {
    "total_processing_time_sec": 44.1,
    "avg_processing_fps": 6.8
  },
  "summary": {
    "total_zone_entry_events": 2,
    "events_by_class": {"person": 2},
    "unique_objects_seen": {"person": 8, "car": 3},
    "total_detections_across_frames": {"person": 1170, "car": 662},
    "objects_already_inside_zone_at_start": 2
  },
  "events": [
    {
      "event_id": 1,
      "event_type": "zone_entry",
      "zone": "restricted_zone_1",
      "timestamp_sec": 2.2,
      "frame_number": 66,
      "track_id": 2,
      "object_class": "person",
      "confidence": 0.87,
      "bbox_pixels": {"x1": 640, "y1": 210, "x2": 720, "y2": 400},
      "foot_point_pixels": {"x": 680, "y": 400}
    }
  ]
}
```

### `video_info`: about the input video

| Field | Meaning |
|---|---|
| `source` | Video file path (or webcam index) used as input |
| `resolution.width / height` | Frame size in pixels |
| `source_fps` | Frame rate of the original video |
| `frames_processed` | Number of frames actually analysed |
| `duration_processed_sec` | Length of video analysed, in seconds (`frames_processed / source_fps`) |

### `config`: settings used for this run

| Field | Meaning |
|---|---|
| `model` | YOLO weights used |
| `confidence_threshold` | Detections below this confidence were discarded |
| `event_cooldown_sec` | Minimum gap between two events for the same object |
| `warmup_frames` | Initial frames in which objects already inside the zone are ignored for entry events |
| `zone.name` | Label of the restricted zone |
| `zone.polygon_pixels` | Zone corners in pixel coordinates `[x, y]` |

### `performance`: processing speed

| Field | Meaning |
|---|---|
| `total_processing_time_sec` | Wall-clock time taken to process the video |
| `avg_processing_fps` | Frames processed per second on this machine. Real-time needs a value at or above `source_fps` |

### `summary`: quick overview of the run

| Field | Meaning |
|---|---|
| `total_zone_entry_events` | Number of entries into the restricted zone |
| `events_by_class` | Entry events split by object type |
| `unique_objects_seen` | Number of **distinct** objects per class (distinct track IDs) |
| `total_detections_across_frames` | Sum of detections over all frames. One person visible for 50 frames counts 50 times, so this is much larger than the number of unique objects |
| `objects_already_inside_zone_at_start` | Objects that were in the zone from the start and were therefore not counted as entries |

### `events`: one entry per zone entry

| Field | Meaning |
|---|---|
| `event_id` | Running event number, starting from 1 |
| `event_type` | Kind of event. Currently only `zone_entry` |
| `zone` | Name of the zone that was entered |
| `timestamp_sec` | Time in the video when the entry happened |
| `frame_number` | Frame index of the entry (starts at 0) |
| `track_id` | Persistent ID of the object assigned by the tracker |
| `object_class` | What the object is, e.g. `person`, `car` |
| `confidence` | Detector confidence for that frame (0 to 1) |
| `bbox_pixels` | Bounding box corners: top-left `(x1, y1)`, bottom-right `(x2, y2)` |
| `foot_point_pixels` | Bottom-center point of the box, the point tested against the zone |

## Limitations

- Zone is a single fixed polygon defined by the user.
- The zone test uses the box's bottom-center, which assumes a roughly front or top-down camera angle.
- Track IDs can switch when objects are heavily occluded.
- Speed depends on hardware. A CPU run is slower than real time; use a GPU or a smaller input for live streams.

## Possible improvements

- Multiple zones and zone-exit / loitering events
- RTSP stream input for live CCTV cameras
- Sending events to a message queue or webhook
- Interactive zone drawing

## Tech stack

Python, OpenCV, Ultralytics YOLO, NumPy, JSON
