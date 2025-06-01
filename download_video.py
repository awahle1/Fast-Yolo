import urllib.request
import os

# Download a sample video (this is a creative commons traffic video)
video_url = "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4"
if not os.path.exists("demo_video.mp4"):
    urllib.request.urlretrieve(video_url, "demo_video.mp4")