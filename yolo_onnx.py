import torch
import cv2
import time
from ultralytics import YOLO
import numpy as np
import collections
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

fps_window = 10

def benchmark_and_save_video(in_path, out_path, session):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {in_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30     # fallback if missing
    out = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        orig_fps,
        (w, h),
    )

    timer = collections.deque(maxlen=fps_window)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.perf_counter()
        results = session.run(None, {"images": frame})
        timer.append(time.perf_counter() - start)

        res = results[0] 
        annotated = res.plot()

        if timer:
            fps = len(timer) / sum(timer)
            cv2.putText(
                annotated,
                f"FPS: {fps:0.1f}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        out.write(annotated)

    cap.release()
    out.release()
    print(f"Finished. Saved to {out_path}")

def main():
    model = YOLO("yolov8x.pt")
    model.export(format='onnx', dynamic=True, opset=12)

    quantize_dynamic(
        model_input='yolov8x.onnx',
        model_output='yolov8x_dynamic.onnx',
        weight_type=QuantType.QInt8,
    )

    session = ort.InferenceSession("yolov8x_dynamic.onnx", providers=["CUDAExecutionProvider"])


    benchmark_and_save_video("data/demo_video.mp4", "outputs/baseline/out_x_onnx.mp4", session)



if __name__ == "__main__":
    # torch.set_float32_matmul_precision("high")  # a small speed gain on Ampere+
    main()