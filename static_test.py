import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

def check_file(image):
    if not os.path.isfile(image):
        print(f"Unable to find {image}")
        exit(-1)

model = "best.pt"
check_file(model)
model = YOLO(f"{model}")

image = "test_1.jpeg"
check_file(image)

# Run detection
results = model(image)

# Annotate and display the image
annotated_image = results[0].plot()  # Automatically plots the detections on the image

# Use matplotlib to display the annotated image
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.axis('on')  # Turn off axis
plt.title("Detections")
plt.show()
