from ultralytics import YOLO
import cv2
import numpy as np
from picamera2 import Picamera2
import time

MODEL_PATH = "best.pt"

# Load YOLO model and optimize for inference
yolo_model = YOLO(MODEL_PATH)
yolo_model.fuse()  # Fuse model layers for faster inference

SIZE = (640, 480)
# Initialize Picamera2 with optimized settings
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(
   main={"format": 'RGB888',
         "size": SIZE,
         "stride": 32},  # Align stride for better performance
   buffer_count=2  # Reduce buffer count for lower latency
)
picam2.configure(preview_config)

# Set camera parameters for speed
picam2.set_controls({"FrameDurationLimits": (16666, 16666)})  # ~60fps
picam2.start()

# Warm up camera
time.sleep(1)

try:
   while True:
       # Capture frame
       frame = picam2.capture_array()
      
       # Run YOLO inference with optimizations
       results = yolo_model(frame,
                          conf=0.8,  # High confidence threshold
                          verbose=False,  # Disable verbose output
                          stream=True,  # Enable streaming mode
                          device='cpu')  # Specify device explicitly      
       # Process results
       for result in results:
           boxes = result.boxes.xyxy
           if len(boxes):
               # Process all boxes at once using numpy operations
               boxes_np = boxes.cpu().numpy().astype(np.int32)
              
               # Draw object count once
               count_text = f"Objects: {len(boxes_np)}"
               cv2.putText(frame, count_text, (10, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
                         cv2.LINE_AA)
              
               # Draw all boxes efficiently
               for box in boxes_np:
                   x1, y1, x2, y2 = box
                   # Draw rectangle
                   cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                  
                   # Add dimensions text
                   dim_text = f"{x2-x1}x{y2-y1}"
                   cv2.putText(frame, dim_text, (x1, y1-10),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                             cv2.LINE_AA)
           else:
               # Show zero count when no objects detected
               cv2.putText(frame, "Objects: 0", (10, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
                         cv2.LINE_AA)
      
       # Efficient color conversion and display
       frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
       cv2.imshow("Live Detection", frame_bgr)
      
       # Check for quit with minimal delay
       if (cv2.waitKey(1) & 0xFF) == ord('q'):
           break
finally:
   cv2.destroyAllWindows()
   picam2.stop()


