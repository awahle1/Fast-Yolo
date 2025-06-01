import torch
import cv2
import time
from ultralytics import YOLO
import numpy as np

def benchmark_and_save_video(input_video_path, output_video_path, max_frames=None):
    model = YOLO('yolov8l.pt')
    model.to('cuda')
    
    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    fps_in = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input video: {width}x{height} @ {fps_in}fps, {total_frames} frames")
    
    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps_in, (width, height))
    
    # Warmup
    dummy_frame = torch.randn(1, 3, 640, 640).cuda()
    for _ in range(10):
        with torch.no_grad():
            _ = model.model(dummy_frame)
    
    # Benchmark variables
    inference_times = []
    fps_values = []
    frame_count = 0
    total_start_time = time.time()
    
    print("Processing video...")
    
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frame_count >= max_frames):
            break
        
        # Run inference and time it
        start_time = time.time()
        results = model(frame, verbose=False)
        inference_time = time.time() - start_time
        
        # Calculate metrics
        current_fps = 1.0 / inference_time
        inference_times.append(inference_time * 1000)  # Convert to ms
        fps_values.append(current_fps)
        
        # Get annotated frame
        annotated_frame = results[0].plot()
        
        # Add performance overlay
        overlay_height = 120
        overlay = np.zeros((overlay_height, width, 3), dtype=np.uint8)
        overlay[:] = (0, 0, 0)  # Black background
        
        # Current metrics
        cv2.putText(overlay, f'Frame: {frame_count + 1}/{total_frames if not max_frames else max_frames}', 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay, f'Inference FPS: {current_fps:.1f}', 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(overlay, f'Inference Time: {inference_time*1000:.1f}ms', 
                   (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Running averages
        if len(fps_values) >= 30:  # After 30 frames
            avg_fps = np.mean(fps_values[-30:])
            avg_time = np.mean(inference_times[-30:])
            cv2.putText(overlay, f'Avg FPS (30f): {avg_fps:.1f}', 
                       (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(overlay, f'Avg Time (30f): {avg_time:.1f}ms', 
                       (300, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Overall stats
        if frame_count > 0:
            overall_avg_fps = np.mean(fps_values)
            cv2.putText(overlay, f'Overall Avg FPS: {overall_avg_fps:.1f}', 
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 255, 128), 2)
        
        # Combine annotated frame with overlay
        final_frame = np.vstack([annotated_frame, overlay])
        
        # Resize if needed to maintain aspect ratio
        if final_frame.shape[:2] != (height, width):
            final_frame = cv2.resize(final_frame, (width, height + overlay_height))
        
        out.write(final_frame)
        
        frame_count += 1
        if frame_count % 50 == 0:
            elapsed = time.time() - total_start_time
            processing_fps = frame_count / elapsed
            print(f"Processed {frame_count} frames, Processing FPS: {processing_fps:.2f}, Avg Inference FPS: {np.mean(fps_values):.2f}")
    
    # Cleanup
    cap.release()
    out.release()
    
    # Final statistics
    total_time = time.time() - total_start_time
    print(f"\nFinal Results:")
    print(f"Total frames processed: {frame_count}")
    print(f"Average inference FPS: {np.mean(fps_values):.2f}")
    print(f"Min inference FPS: {min(fps_values):.2f}")
    print(f"Max inference FPS: {max(fps_values):.2f}")
    print(f"Average inference time: {np.mean(inference_times):.2f}ms")
    print(f"Total processing time: {total_time:.2f}s")
    print(f"Video saved to: {output_video_path}")
    
    return {
        'avg_fps': np.mean(fps_values),
        'min_fps': min(fps_values),
        'max_fps': max(fps_values),
        'avg_inference_time': np.mean(inference_times),
        'fps_history': fps_values,
        'inference_history': inference_times
    }

# Run the benchmark
baseline_results = benchmark_and_save_video('data/demo_video.mp4', 'outputs/baseline/demo_out.mp4')