"""Real-time object detection with restricted-zone event generation.

Detects and tracks objects with YOLO, flags objects entering a polygonal
restricted zone, and writes a structured JSON report plus an annotated video.
"""
import argparse
import json
import time
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO

ZONE_NAME = "restricted_zone_1"
# Normalized (x, y) pairs: top-left, top-right, bottom-right, bottom-left
DEFAULT_ZONE = "0.55,0.35,0.95,0.35,0.95,0.95,0.55,0.95"
WARMUP_FRAMES = 5  # objects already inside the zone in these first frames are not counted as entries


def parse_args():
    p = argparse.ArgumentParser(description="CCTV object detection and zone-entry events")
    p.add_argument("--video", required=True, help="input video path, or 0 for webcam")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    p.add_argument("--seconds", type=float, default=10, help="seconds of video to process")
    p.add_argument("--zone", default=DEFAULT_ZONE, help="normalized polygon: x1,y1,x2,y2,...")
    p.add_argument("--classes", type=int, nargs="*", default=None,
                   help="COCO class ids to keep, e.g. 0 2 for person and car")
    p.add_argument("--cooldown", type=float, default=1.0,
                   help="min seconds between two events for the same track")
    p.add_argument("--out", default="output.mp4")
    p.add_argument("--events", default="events.json")
    return p.parse_args()


def parse_zone(text, w, h):
    vals = [float(v) for v in text.split(",")]
    pts = zip(vals[0::2], vals[1::2])
    return np.array([(int(x * w), int(y * h)) for x, y in pts], np.int32)


def main():
    args = parse_args()
    src = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(src_fps * args.seconds)
    cooldown_frames = int(src_fps * args.cooldown)

    zone = parse_zone(args.zone, w, h)
    model = YOLO(args.model)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))

    was_inside = {}            # track_id -> inside the zone in the previous frame
    last_event = {}            # track_id -> frame index of its last event
    unique_ids = {}            # class name -> set of track ids seen
    initially_inside = set()   # track ids already in the zone at video start
    events = []
    det_counts = Counter()
    fps_smooth = 0.0
    frame_idx = 0
    t_start = time.time()

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.time()
        res = model.track(frame, persist=True, conf=args.conf,
                          classes=args.classes, verbose=False)[0]
        inst_fps = 1 / max(time.time() - t0, 1e-6)
        fps_smooth = inst_fps if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * inst_fps

        # draw the restricted zone
        overlay = frame.copy()
        cv2.fillPoly(overlay, [zone], (0, 0, 255))
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
        cv2.polylines(frame, [zone], True, (0, 0, 255), 2)

        boxes = res.boxes
        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)

        for box, tid in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            label = model.names[int(box.cls[0])]
            det_counts[label] += 1

            # bottom-center of the box approximates where the object touches the ground
            foot = (int((x1 + x2) / 2), y2)
            inside = cv2.pointPolygonTest(zone, foot, False) >= 0

            if tid is not None:
                unique_ids.setdefault(label, set()).add(tid)

                if frame_idx < WARMUP_FRAMES and inside:
                    initially_inside.add(tid)

                entered = inside and not was_inside.get(tid, False)
                cooled_down = frame_idx - last_event.get(tid, -cooldown_frames) >= cooldown_frames
                if entered and frame_idx >= WARMUP_FRAMES and cooled_down:
                    events.append({
                        "event_id": len(events) + 1,
                        "event_type": "zone_entry",
                        "zone": ZONE_NAME,
                        "timestamp_sec": round(frame_idx / src_fps, 2),
                        "frame_number": frame_idx,
                        "track_id": tid,
                        "object_class": label,
                        "confidence": round(conf, 2),
                        "bbox_pixels": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "foot_point_pixels": {"x": foot[0], "y": foot[1]},
                    })
                    last_event[tid] = frame_idx
                    print(f"[EVENT] {label} #{tid} entered {ZONE_NAME} at {frame_idx / src_fps:.1f}s")
                was_inside[tid] = inside

            color = (0, 0, 255) if inside else (0, 200, 0)
            text = f"{label} {conf:.2f}" + (f" #{tid}" if tid is not None else "")
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(frame, f"FPS: {fps_smooth:.1f}  Events: {len(events)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    elapsed = time.time() - t_start

    report = {
        "video_info": {
            "source": str(args.video),
            "resolution": {"width": w, "height": h},
            "source_fps": round(src_fps, 2),
            "frames_processed": frame_idx,
            "duration_processed_sec": round(frame_idx / src_fps, 2),
        },
        "config": {
            "model": args.model,
            "confidence_threshold": args.conf,
            "event_cooldown_sec": args.cooldown,
            "warmup_frames": WARMUP_FRAMES,
            "zone": {"name": ZONE_NAME, "polygon_pixels": zone.tolist()},
        },
        "performance": {
            "total_processing_time_sec": round(elapsed, 2),
            "avg_processing_fps": round(frame_idx / elapsed, 2),
        },
        "summary": {
            "total_zone_entry_events": len(events),
            "events_by_class": dict(Counter(e["object_class"] for e in events)),
            "unique_objects_seen": {k: len(v) for k, v in unique_ids.items()},
            "total_detections_across_frames": dict(det_counts),
            "objects_already_inside_zone_at_start": len(initially_inside),
        },
        "events": events,
    }

    with open(args.events, "w") as f:
        json.dump(report, f, indent=2)

    print("\n--- Summary ---")
    print(json.dumps(report["summary"], indent=2))
    print(f"Avg processing FPS: {report['performance']['avg_processing_fps']}")
    print(f"Annotated video: {args.out} | Report: {args.events}")


if __name__ == "__main__":
    main()