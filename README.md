🔢 Handwritten Digit Recognition Using CNN

A deep learning-based Handwritten Digit Recognition system built using Convolutional Neural Networks (CNN) and the MNIST dataset. This project allows users to draw or upload handwritten digits and predicts the corresponding number with high accuracy through an interactive Gradio interface.

🚀 Features
✍️ Draw or upload handwritten digit images
🧠 CNN model trained on the MNIST dataset
🎯 Predicts digits from 0 to 9
📊 Displays prediction confidence
🌐 Interactive Gradio web interface
☁️ Deployable on Hugging Face Spaces
⚡ High accuracy (~99%)
🛠️ Technologies Used
Python
TensorFlow / Keras
Convolutional Neural Networks (CNN)
NumPy
OpenCV
Gradio
MNIST Dataset
📂 Dataset

The model is trained on the MNIST dataset, consisting of 70,000 grayscale images of handwritten digits (28×28 pixels).

Training samples: 60,000
Testing samples: 10,000
Classes: 10 (Digits 0–9)
🧠 Model Architecture
Conv2D Layer (32 filters)
MaxPooling2D
Conv2D Layer (64 filters)
MaxPooling2D
Flatten Layer
Dense Layer (128 neurons)
Dropout Layer
Output Layer (10 classes with Softmax)
📈 Performance
Training Accuracy: ~99%
Test Accuracy: ~98–99%
