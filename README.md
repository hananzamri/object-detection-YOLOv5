# Object Detection Using YOLOv5

## Overview

This project implements an Object Detection System using the YOLOv5s (You Only Look Once Version 5 Small) model. Users can upload an image, detect objects automatically, visualize bounding boxes with confidence scores, and download the annotated output image.

The project is designed to run on Google Colab using PyTorch, OpenCV, and a pretrained YOLOv5 model.

---

## Features

- Upload an image directly in Google Colab
- Detect multiple objects in a single image
- Display object names and confidence scores
- Draw bounding boxes around detected objects
- Save detection results as an image
- Download the annotated output image
- GPU acceleration support
- Uses pretrained YOLOv5s model trained on the COCO dataset

---

## Technologies Used

- Python
- PyTorch
- YOLOv5
- OpenCV
- NumPy
- Google Colab

---

## Project Workflow

```text
Image Upload
     │
     ▼
Load YOLOv5 Model
     │
     ▼
Object Detection
     │
     ▼
Extract Results
     │
     ▼
Display Labels &
Confidence Scores
     │
     ▼
Draw Bounding Boxes
     │
     ▼
Save Output Image
     │
     ▼
Download Result
```

---

## Installation

Install the required dependencies:

```python
!pip install ultralytics -q
!pip install opencv-python -q
!rm -rf /root/.cache/torch/hub/ultralytics_yolov5_master
```

---

## Required Libraries

```python
import numpy as np
import cv2
import torch
from google.colab.patches import cv2_imshow
from google.colab import files
```

---

## Model Loading

The project loads a pretrained YOLOv5s model from the official YOLOv5 repository.

```python
yolov5 = torch.hub.load(
    'ultralytics/yolov5',
    'yolov5s',
    pretrained=True,
    trust_repo=True
)
```

Set the confidence threshold:

```python
yolov5.conf = 0.30
```

Only detections with confidence greater than 30% will be displayed.

---

## Usage

### Step 1: Check GPU Availability

```python
print("GPU Available:", torch.cuda.is_available())
!nvidia-smi
```

### Step 2: Upload an Image

```python
uploaded = files.upload()
image_path = list(uploaded.keys())[0]
```

### Step 3: Read and Display the Image

```python
image = cv2.imread(image_path)
cv2_imshow(image)
```

### Step 4: Run Object Detection

```python
results = yolov5(image)
```

### Step 5: Display Detection Results

```python
objects = results.xyxyn[0].detach().cpu().numpy()

for obj in objects:
    if obj[-2] > 0.30:
        class_name = classes[int(obj[-1])]
        confidence = obj[-2]
        print(f"{class_name}: {confidence:.2f}")
```

Example Output:

```text
person: 0.98
car: 0.91
dog: 0.86
```

### Step 6: Draw Bounding Boxes

```python
cv2.rectangle(
    output_image,
    tuple(tl[0]),
    tuple(br),
    (0, 0, 255),
    2
)
```

Labels and confidence scores are displayed above each detected object.

### Step 7: Save and Download Result

```python
cv2.imwrite("output_result.png", output_image)
files.download("output_result.png")
```

---

## Sample Output

### Detection Results

```text
person: 0.98
car: 0.92
bicycle: 0.88

------------------------------
Total detected objects: 3
Unique object types: ['person', 'car', 'bicycle']
```

### Generated File

```text
output_result.png
```

The output image contains:

- Bounding boxes
- Object labels
- Confidence scores

---

## Supported Object Classes

The pretrained YOLOv5s model is trained on the COCO dataset and can detect 80 object categories, including:

- Person
- Bicycle
- Car
- Motorcycle
- Bus
- Truck
- Traffic Light
- Dog
- Cat
- Bird
- Horse
- Laptop
- Keyboard
- Cell Phone
- Bottle
- Chair
- Dining Table

and many more.

---

## Project Structure

```text
project/
│
├── object_detection.ipynb
├── input_image.jpg
├── output_result.png
└── README.md
```

---

## Output

| File | Description |
|--------|-------------|
| output_result.png | Image containing detected objects with bounding boxes and labels |

---

## Future Improvements

- Real-time webcam detection
- Video object detection
- Custom dataset training
- Object counting
- Object tracking
- Streamlit web application
- YOLOv8 integration
- Detection analytics dashboard

---

## Conclusion

This project demonstrates how to use a pretrained YOLOv5 model for object detection in images. By combining PyTorch, OpenCV, and Google Colab, users can quickly perform object detection, visualize results, and export annotated images with minimal setup.

---

## License

MIT License
