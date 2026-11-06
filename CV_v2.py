# Real-Time CCTV Object Detection & Zone-Entry Event Pipeline

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8n-00FFFF)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)
![Output](https://img.shields.io/badge/Output-JSON%20events-success)

A video-analytics prototype that **detects and tracks objects in CCTV-style footage** with YOLOv8 + OpenCV, and emits **structured JSON events** whenever a tracked object enters a user-defined **restricted zone**.

Instead of just drawing boxes, the pipeline turns pixels into *machine-readable events* (who, what, when, where, how confident) that a downstream system can consume: alerts, dashboards, queues, or audit logs.

![Restricted-zone detection demo: tracked objects with IDs, red restricted zone, live FPS and event counter](output/demo.png)

---

## Results

Measured on the included `test.mp4` (first 10 seconds), default settings (`yolov8n.pt`, confidence 0.4):

| Metric | Value |
|---|---|
| Input resolution | 922 x 524 |
| Source frame rate | 30.06 FPS |
| Frames processed | 300 (9.98 s of video) |
| Hardware | HP 15s laptop |
| Total processing time | 37.31 s |
| Average processing speed (end-to-end) | **8.04 FPS** (about 0.27x real-time) |
| Detections across all frames | 1,832 (1,170 person, 662 car) |
| Unique tracked objects | 12 (9 person, 3 car) |
| Zone-entry events raised | **2** (both `person`) |
| Objects already inside zone at start (excluded) | 2 |
| Restricted zone size | 368 x 314 px, about 24% of the frame |

> The reported FPS is **end-to-end** (frames processed divided by wall-clock time, including video decode, drawing and writing the output video), not inference-only. The FPS shown on the video overlay is a smoothed inference-only value, so it reads higher.

> **Reading the numbers:** 1,832 detections collapse to 12 unique objects because each object is detected in many frames. Tracking turns detections into *identities*, and the zone logic turns identities into *events*: 1,832 detections became 2 events.

---

## How it works

```mermaid
flowchart LR
    A[Video / Webcam] --> B[YOLOv8n detect + track]
    B --> C{conf >= 0.4?}
    C -- no --> X[drop]
    C -- yes --> D[Foot point = bottom-center of box]
    D --> E{Was outside, now inside zone?}
    E -- no --> F[Update state]
    E -- yes --> G{Warm-up passed and cooldown elapsed?}
    G -- no --> F
    G -- yes --> H[Emit zone_entry event]
    H --> I[(events.json)]
    B --> J[Annotated output.mp4]
```

1. Read the video and process the first `--seconds` seconds.
2. Run YOLOv8 with Ultralytics' built-in tracker (`model.track`, `persist=True`) on each frame; discard detections below `--conf`.
3. For every tracked object, take the **bottom-center point** of its bounding box (roughly where it touches the ground).
4. Test whether that point lies inside the restricted polygon.
5. If the object was **outside** in the previous frame and is **inside** now, log a `zone_entry` event.
6. Write the annotated video and the JSON report.

### Two safeguards that keep events clean

- **Warm-up (5 frames):** objects already inside the zone at the start are *not* counted as entries. They are reported separately as `objects_already_inside_zone_at_start`.
- **Cooldown (1.0 s default):** the same `track_id` cannot fire again within `--cooldown` seconds (converted to frames using the source FPS, so 30 frames on the sample clip), so flickering on the zone border does not create duplicate events.

---

## Features

- Frame-by-frame detection with **YOLOv8 (Ultralytics)**
- Confidence-based filtering (`--conf`)
- **Multi-object tracking**: every object keeps a persistent `track_id`
- Bounding boxes annotated with class, confidence and track ID
- **Live FPS overlay** on the output video
- **Polygon restricted-zone** entry detection (foot-point test)
- Class filtering using COCO IDs (e.g. persons and cars only)
- Webcam support (`--video 0`)
- Structured **JSON report**: video info, config, performance, summary and events
- Duplicate-event protection (warm-up + per-track cooldown)

---

## Quick start

```bash
git clone https://github.com/Krish033-source/OpenCV_Yolo.git
cd OpenCV_Yolo
pip install -r requirements.txt

python CV_v2.py --video test.mp4 --seconds 10 --out output/output.mp4 --events output/events.json
```

The YOLO weights (`yolov8n.pt`) are downloaded automatically on the first run.

### More examples

```bash
# Persons only, custom zone covering the left half of the frame
python CV_v2.py --video test.mp4 --classes 0 \
  --zone "0.05,0.3,0.5,0.3,0.5,0.95,0.05,0.95"

# Persons and cars, stricter confidence, longer cooldown
python CV_v2.py --video test.mp4 --classes 0 2 --conf 0.5 --cooldown 2.0

# Live webcam
python CV_v2.py --video 0 --seconds 30
```

### CLI arguments

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

Zone coordinates are **normalized (0 to 1)**, so the same zone works on any resolution. The default zone resolves to `[507,183] to [875,497]` pixels on a 922 x 524 frame.

---

## Output

- `output.mp4`: video with the zone (red), bounding boxes, labels, FPS and an event counter. Objects inside the zone turn red.
- `events.json`: full machine-readable report. A real run is checked in at [`output/events.json`](output/events.json).

### JSON report structure

```json
{
  "video_info":  { "source": "test.mp4", "resolution": {"width": 922, "height": 524},
                   "source_fps": 30.06, "frames_processed": 300, "duration_processed_sec": 9.98 },
  "config":      { "model": "yolov8n.pt", "confidence_threshold": 0.4, "event_cooldown_sec": 1.0,
                   "warmup_frames": 5,
                   "zone": { "name": "restricted_zone_1",
                             "polygon_pixels": [[507,183],[875,183],[875,497],[507,497]] } },
  "performance": { "total_processing_time_sec": 37.31, "avg_processing_fps": 8.04 },
  "summary":     { "total_zone_entry_events": 2,
                   "events_by_class": {"person": 2},
                   "unique_objects_seen": {"car": 3, "person": 9},
                   "total_detections_across_frames": {"car": 662, "person": 1170},
                   "objects_already_inside_zone_at_start": 2 },
  "events":      [ ... ]
}
```

### A real event

```json
{
  "event_id": 1,
  "event_type": "zone_entry",
  "zone": "restricted_zone_1",
  "timestamp_sec": 0.83,
  "frame_number": 25,
  "track_id": 3,
  "object_class": "person",
  "confidence": 0.85,
  "bbox_pixels": {"x1": 479, "y1": 56, "x2": 564, "y2": 259},
  "foot_point_pixels": {"x": 521, "y": 259}
}
```

### Field reference

| Section | Field | Meaning |
|---|---|---|
| `video_info` | `frames_processed`, `duration_processed_sec` | Frames analysed and their length in seconds |
| `config` | `warmup_frames`, `event_cooldown_sec` | Settings used for duplicate-event protection |
| `performance` | `avg_processing_fps` | Frames per second on this machine; real-time needs a value at or above `source_fps` |
| `summary` | `unique_objects_seen` | Number of **distinct** objects per class (distinct track IDs) |
| `summary` | `total_detections_across_frames` | Sum of detections over all frames; one person seen for 50 frames counts 50 times |
| `summary` | `objects_already_inside_zone_at_start` | Objects inside the zone from the start, not counted as entries |
| `events[]` | `track_id` | Persistent ID assigned by the tracker |
| `events[]` | `foot_point_pixels` | Bottom-center of the box, the exact point tested against the zone |

---

## Limitations

- **Speed:** 8.04 FPS end-to-end on an HP 15s laptop is about 3.7x slower than real time (37.31 s to process 9.98 s of video). Use a GPU, a smaller input size, or frame skipping for live streams.
- A single fixed polygon zone, defined by the user.
- The foot-point test assumes a roughly front or top-down camera angle.
- Track IDs can switch under heavy occlusion, and an object that is first detected *inside* the zone after the warm-up window (or re-acquired under a new ID) is counted as an entry.
- Evaluated on one short clip; detection and tracking accuracy (precision/recall, ID switches) has not been formally measured.

## Roadmap

- [ ] Multiple zones, plus zone-exit and loitering events
- [ ] RTSP stream input for live CCTV cameras
- [ ] Push events to a message queue or webhook
- [ ] Interactive zone drawing
- [ ] Benchmark on GPU and with a larger model (`yolov8s`, `yolov8m`)
- [ ] Unit tests for the zone-entry logic

## Tech stack

Python, OpenCV, Ultralytics YOLOv8, NumPy, JSON

## Repository structure

```
OpenCV_Yolo/
├── CV_v2.py            # detection, tracking and zone-event pipeline
├── requirements.txt
├── test.mp4            # sample footage
├── output/
│   ├── events.json     # sample report from a real run
│   └── demo.png        # screenshot of the annotated output
└── README.md
```

## Author

**Krishna Lal** · [GitHub](https://github.com/Krish033-source) · [LinkedIn](https://www.linkedin.com/in/krishna-lal-5b1a22321) · [Portfolio](https://krbtech.xyz)
