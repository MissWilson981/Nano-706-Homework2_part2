**Overview**

This notebook explores machine learning and computer vision through two homework tasks:
1. Handwritten-digit classification: Classify MNIST digits using Random Forest, Logistic Regression, and a Multilayer Perceptron (MLP).
2. Image detection using YOLO: Apply a pretrained YOLOv5s model to a personal photograph.

**Part 1: MNIST Classification**

The MNIST dataset consists of grayscale images of handwritten digits from 0 to 9. The homework loads the dataset with fetch_openml('mnist_784', version=1), normalizes pixel values by dividing by 255, and uses an 80% training / 20% testing split (random_state=42).

**Model Accuracy Comparison**

| Machine Learning Model | Accuracy (%) |
|---|---:|
| Logistic Regression | 92.02 |
| Random Forest | 96.75 |
| Multilayer Perceptron (MLP) | 97.55 |

<img width="708" height="451" alt="MNIST classificarion" src="https://github.com/user-attachments/assets/cb695983-1d2c-49be-ad6f-fae70b9e8066" />

<img width="950" height="202" alt="MNIST predictions" src="https://github.com/user-attachments/assets/26f9ae20-99d0-4e49-b185-d074c313346c" />

The MLP achieved the highest accuracy among the three models in this run.

**Part 2: YOLO Image Detection**

The homework loads a pretrained YOLOv5s model through PyTorch Hub and runs inference on an uploaded photograph using results = model(img). The output is displayed using results.show().

<img width="600" height="600" alt="My face" src="https://github.com/user-attachments/assets/cd87536e-10fd-4ead-befa-f5c76f498be8" />

The standard YOLOv5s model detects object classes such as person and car. It does not contain a dedicated face class. Therefore, a detected person bounding box can not be described as face-specific detection. A YOLO model trained for face detection would be needed to draw a face-specific bounding box.
