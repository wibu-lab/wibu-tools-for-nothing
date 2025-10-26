"""Small utility script to test OpenCV + GStreamer camera capture and measure FPS.

This script was written to help verify that a camera pipeline works with OpenCV
and GStreamer, measure the effective frame rate, and save a sample image for
each tested resolution.

Main features:
- Iterate a short list of common/reserved resolutions (modifiable at the top)
- Try opening a GStreamer pipeline per resolution
- Capture N frames and compute average FPS (measured from successful reads)
- Save the last successfully captured frame with an overlay showing AVG FPS

Usage examples:
  python fov_test_opencv_gstreamer.py --frames 120 --save-dir ./data
  python fov_test_opencv_gstreamer.py --single --pipeline "<your pipeline>"

Note: this script purposely keeps dependencies minimal (only OpenCV). It
assumes OpenCV was built with GStreamer support. On failure, it prints build
info to help debug missing GStreamer support.
"""
import argparse
import time
import os
import cv2


# A small set of resolutions to test. Add or remove as needed.
# (You can include the camera's maximum resolution if desired.)
resolution_list = [
    (1640, 1232),
    (1540, 1232),
    (1280, 720),
    (1920, 1080),
    (800, 480),
]


# Default one-shot pipeline used when --single is provided. Kept as a constant
# so users can override it from the CLI without editing the file.
GSTREAMER_PIPELINE = (
    "libcamerasrc ! video/x-raw,width=3280,height=2464,format=RGB "
    "! videoconvert ! videoscale ! appsink"
)


def measure_fps(cap, num_frames=20, save_dir="./data", display=False, resolution=None):
    """Capture up to `num_frames` frames from `cap` and measure FPS.

    Behavior and return values:
    - Only timestamps for successfully read frames are used to compute FPS.
    - The function records the last successfully captured frame and saves it to
      `save_dir` (one image per call) with an overlay showing the average FPS.

    Args:
        cap (cv2.VideoCapture): An opened capture object (GStreamer pipeline).
        num_frames (int): Maximum number of capture attempts.
        save_dir (str): Directory to save the sample image.
        display (bool): If True, show frames in a window during capture.
        resolution (tuple|None): Optional (width, height) used for filename/overlay.

    Returns:
        dict: {"total_frames": int, "saved": 0|1, "elapsed": float,
               "avg_fps": float, "timestamps": list}
    """
    os.makedirs(save_dir, exist_ok=True)

    timestamps = []          # per-successful-frame timestamps (seconds)
    saved = 0
    last_frame = None
    frames_captured = 0

    prev_ts = None
    start = time.time()

    for i in range(num_frames):
        # Attempt to read a frame and measure the wall-clock time of the read.
        ret, frame = cap.read()
        now = time.time()

        if not ret or frame is None:
            # Helpful debug line: indicates a transient or persistent capture issue.
            print(f"Frame {i+1}: capture failed (ret={ret}).")
            # Continue trying until we have attempted `num_frames` reads.
            continue

        # Successful read: record timestamp and compute instantaneous FPS.
        timestamps.append(now)
        frames_captured += 1

        if prev_ts is None:
            inst_fps = 0.0
        else:
            dt = now - prev_ts
            inst_fps = 1.0 / dt if dt > 0 else 0.0
        prev_ts = now

        # Overlay instantaneous FPS and optional resolution string for debugging.
        if resolution:
            text = f"FPS: {inst_fps:.2f}; {resolution[0]}x{resolution[1]}"
        else:
            text = f"FPS: {inst_fps:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Keep the last successful frame (we'll save only one at the end).
        last_frame = frame.copy()

        # Optionally display while capturing. Press ESC to abort early.
        if display:
            cv2.imshow("capture", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                print("User requested exit.")
                break

    end = time.time()
    elapsed = end - start

    # Use frames_captured (only successful reads) to compute average FPS.
    total_frames = frames_captured
    avg_fps = total_frames / elapsed if elapsed > 0 else 0.0

    # Save only the last successful frame, annotated with average FPS.
    if last_frame is not None:
        overlay_text = f"AVG FPS: {avg_fps:.2f}"
        cv2.putText(last_frame, overlay_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        if resolution:
            fname = os.path.join(save_dir, f"captured_image_last_{resolution[0]}x{resolution[1]}.jpg")
        else:
            fname = os.path.join(save_dir, "captured_image_last.jpg")
        cv2.imwrite(fname, last_frame)
        saved = 1

    if display:
        cv2.destroyAllWindows()

    return {
        "total_frames": total_frames,
        "saved": saved,
        "elapsed": elapsed,
        "avg_fps": avg_fps,
        "timestamps": timestamps,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Capture from camera and measure FPS using GStreamer pipeline"
    )

    # Allow some configurable options useful when publishing this script.
    parser.add_argument("--pipeline", default=GSTREAMER_PIPELINE, help="GStreamer pipeline to open the camera")
    parser.add_argument("--frames", type=int, default=120, help="Number of frames to attempt for FPS measurement")
    parser.add_argument("--save-dir", default="./data", help="Directory to save captured frames")
    parser.add_argument("--display", action="store_true", help="Show frames in a window while capturing")
    parser.add_argument("--single", action="store_true", help="Use the provided pipeline only once instead of trying all resolutions")
    parser.add_argument("--format", default="YUY2", help="Pixel format to request when testing resolutions (default: YUY2)")
    parser.add_argument("--framerate", default="12/1", help="Framerate string for pipeline (e.g. '30/1', default: 12/1)")
    args = parser.parse_args()

    # If user wants to use a single pipeline exactly as provided, keep the original behavior.
    if args.single:
        cap = cv2.VideoCapture(args.pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            print("Error: Could not open camera with GStreamer pipeline.")
            print("Pipeline used:", args.pipeline)
            print(cv2.getBuildInformation())
            return 1

        print(f"Starting capture for {args.frames} frames using provided pipeline...")
        result = measure_fps(cap, num_frames=args.frames, save_dir=args.save_dir, display=args.display)
        cap.release()

        print("\nCapture summary:")
        print(f"  Total frames read: {result['total_frames']}")
        print(f"  Frames saved: {result['saved']}")
        print(f"  Elapsed time: {result['elapsed']:.3f} s")
        print(f"  Average FPS: {result['avg_fps']:.2f}")
        return 0

    # Otherwise, iterate through the predefined resolution_list and capture for each
    for (w, h) in resolution_list:
        pipeline = (
            f"libcamerasrc ! video/x-raw,width={w},height={h},format={args.format},framerate={args.framerate} "
            "! videoconvert ! appsink"
        )
        print(f"\nTrying resolution {w}x{h} with pipeline: {pipeline}")

        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            print(f"  Warning: Could not open camera for resolution {w}x{h}. Skipping.")
            continue

        # Save frames into the requested save directory. Filenames include resolution.
        res_save_dir = os.path.abspath(args.save_dir)
        print(f"  Saving frames to: {res_save_dir}")

        result = measure_fps(cap, num_frames=args.frames, save_dir=res_save_dir, display=args.display, resolution=(w, h))

        cap.release()

        print("  Capture summary:")
        print(f"    Total frames read: {result['total_frames']}")
        print(f"    Frames saved: {result['saved']}")
        print(f"    Elapsed time: {result['elapsed']:.3f} s")
        print(f"    Average FPS: {result['avg_fps']:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
