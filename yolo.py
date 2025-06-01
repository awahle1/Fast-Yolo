import torch
from ultralytics import YOLO
import time
import cv2
import numpy as np

model = YOLO('yolov8l.pt')
model = model.to('cuda')

model.model.eval()

results = model('path_to_your_test_image.jpg')