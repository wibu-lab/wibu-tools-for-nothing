# fov_test_opencv_gstreamer

Small utility to test OpenCV + GStreamer camera capture and measure effective FPS.

Purpose
- Verify a GStreamer pipeline works with OpenCV's VideoCapture.
- Measure capture performance (average FPS) using successful frame reads.
- Save a sample image per tested resolution.

Requirements
- Python 3.8+ recommended
- OpenCV with GStreamer support (cv2 compiled with GStreamer)

Quick usage

Run the default resolution sweep (creates `./data` and saves a sample image for each resolution):

```bash
python fov_test_opencv_gstreamer.py --frames 120 --save-dir ./data
```

Test a single custom pipeline (one-shot):

```bash
python fov_test_opencv_gstreamer.py --single --pipeline "libcamerasrc ! video/x-raw,width=1280,height=720,format=YUY2,framerate=30/1 ! videoconvert ! appsink" --frames 60
```

Options
- `--frames`: maximum number of capture attempts per test (default: 120)
- `--save-dir`: directory where sample images are written
- `--display`: show frames during capture (press ESC to abort)
- `--format`: pixel format used when testing the resolution sweep (default: YUY2)
- `--framerate`: framerate string used in the pipeline (default: 12/1)
