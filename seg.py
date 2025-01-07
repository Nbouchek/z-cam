from ultralytics import YOLO, SAM
import cv2
import numpy as np
from picamera2 import Picamera2
import time


MODEL_PATH = "best.pt"
SAM_MODEL_PATH = "sam2_b.pt"


# Load models
yolo_model = YOLO(MODEL_PATH)
sam_model = SAM(SAM_MODEL_PATH)


# Initialize Picamera2
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(
   main={"format": 'RGB888', "size": (640, 480)}  # Configure for direct RGB output
)
picam2.configure(preview_config)
picam2.start()


# Give camera time to warm up
time.sleep(2)


try:
   while True:
       # Capture frame from Picamera2 - it will now be in RGB format
       frame = picam2.capture_array()
      
       # Run YOLO with confidence threshold
       results = yolo_model(frame, conf=0.8)  # Only detect objects with 80% or higher confidence


       for result in results:
           boxes = result.boxes.xyxy
           class_ids = result.boxes.cls.int().tolist()
           if len(class_ids):
               # Convert boxes to the format expected by SAM
               boxes_np = boxes.cpu().numpy()
              
               # Process SAM results for all boxes at once
               sam_results = sam_model(result.orig_img, bboxes=boxes_np, verbose=False, save=False, device="cpu")
              
               # Create a single mask for all segmentations
               combined_mask = np.zeros_like(frame[:,:,0], dtype=np.uint8)
               all_contours = []
              
               # Combine all masks and get contours
               for sam_result in sam_results:
                   if sam_result.masks is not None:
                       for mask in sam_result.masks.data:
                           mask_np = mask.cpu().numpy().astype(np.uint8)
                           combined_mask = cv2.bitwise_or(combined_mask, mask_np)
                           contours, _ = cv2.findContours(mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                           all_contours.extend(contours)
              
               # Create and apply single overlay
               if np.any(combined_mask):
                   overlay = frame.copy()
                   overlay[combined_mask > 0] = [144, 238, 144]  # Light green color
                   cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                  
                   # Draw contours and their bounding boxes
                   for contour in all_contours:
                       # Draw contour
                       cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                      
                       # Get and draw bounding box from contour
                       x, y, w, h = cv2.boundingRect(contour)
                       cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                      
                       # Display dimensions
                       dim_text = f"{w}x{h}"
                       cv2.putText(frame, dim_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
      
       # Convert to BGR for display with OpenCV
       frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
       cv2.imshow("Live Camera with YOLO and SAM", frame_bgr)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break


finally:
   # Cleanup
   cv2.destroyAllWindows()
   picam2.stop()