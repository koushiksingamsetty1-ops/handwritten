import gradio as gr
import numpy as np
import cv2
import pickle

# Load model
with open("mnist_cnn_model.pkl", "rb") as file:
    model = pickle.load(file)


def predict_digit(image):
    if image is None:
        return "Please upload or draw a digit!"

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize to 28x28
    img = cv2.resize(gray, (28, 28))

    # Invert colors (white digit on black background)
    img = 255 - img

    # Normalize
    img = img / 255.0

    # Reshape
    img = img.reshape(1, 28, 28, 1)

    # Prediction
    pred = model.predict(img, verbose=0)
    digit = np.argmax(pred)
    confidence = np.max(pred) * 100

    return f"🎯 Predicted Digit: {digit}\n📊 Confidence: {confidence:.2f}%"


custom_css = """
body{
background: linear-gradient(135deg,#0f172a,#1e293b);
}
.gradio-container{
max-width:900px !important;
margin:auto;
border-radius:20px;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
# 🔢 Handwritten Digit Recognition
### 🧠 CNN + MNIST Dataset

Draw a digit ✍️ or Upload an image 📷

""")

    with gr.Row():

        with gr.Column():
            image_input = gr.Image(
                type="numpy",
                label="🖌 Draw or Upload Digit"
            )

            predict_btn = gr.Button(
                "🚀 Predict Digit",
                variant="primary"
            )

        with gr.Column():
            output = gr.Textbox(
                label="Prediction Result",
                lines=3
            )

    gr.Examples(
        examples=[],
        inputs=image_input
    )

    predict_btn.click(
        predict_digit,
        inputs=image_input,
        outputs=output
    )

    gr.Markdown("""
---
### ⚡ Features
✅ CNN Model  
✅ MNIST Dataset  
✅ Draw or Upload Digits  
✅ Confidence Score  
✅ Modern UI
""")

demo.launch()