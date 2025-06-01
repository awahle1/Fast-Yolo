import torch
import cv2
import time
from ultralytics import YOLO
import numpy as np

def benchmark_yolo_realtime():
    # Load model
    model = YOLO('yolov8l.pt')
    model.to('cuda')
    
    # Video source (0 for webcam or path to video file)
    cap = cv2.VideoCapture(0)  # or 'path/to/video.mp4'
    
    # Warmup
    dummy_frame = torch.randn(1, 3, 640, 640).cuda()
    for _ in range(10):
        with torch.no_grad():
            _ = model.model(dummy_frame)
    
    fps_list = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        start_time = time.time()
        
        # Run inference
        results = model(frame, verbose=False)
        
        # Calculate FPS
        inference_time = time.time() - start_time
        fps = 1.0 / inference_time
        fps_list.append(fps)
        
        # Draw results
        annotated_frame = results[0].plot()
        
        # Display FPS on frame
        cv2.putText(annotated_frame, f'FPS: {fps:.1f}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('YOLOv8 Tracking', annotated_frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            avg_fps = np.mean(fps_list[-100:])
            print(f"Average FPS (last 100 frames): {avg_fps:.2f}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return np.mean(fps_list)

# Run baseline
baseline_fps = benchmark_yolo_realtime()
print(f"Baseline FPS: {baseline_fps:.2f}")