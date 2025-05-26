import os
import gradio as gr
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms
import torch
import uuid

# Load pretrained ResNet18 model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Recognition logic
def recognize(uploaded_image):
    if uploaded_image is None:
        return "", gr.update(visible=False)

    input_tensor = transform(uploaded_image).unsqueeze(0)
    with torch.no_grad():
        uploaded_features = model(input_tensor).detach().cpu().numpy()

    best_match = None
    best_score = 0

    for label_folder in os.listdir("dataset"):
        label_path = os.path.join("dataset", label_folder)
        if not os.path.isdir(label_path):
            continue

        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            try:
                img = Image.open(img_path).convert("RGB")
                input_tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    features = model(input_tensor).detach().cpu().numpy()
                score = cosine_similarity(uploaded_features, features)[0][0]
                if score > best_score:
                    best_score = score
                    best_match = label_folder
            except Exception as e:
                print(f"⚠️ Error reading {img_path}: {e}")
                continue

    if best_score >= 0.65:
        prediction = f"🔍 Match: {best_match}"
        confidence = round(best_score * 100, 2)
    else:
        prediction = "❌ Not in dataset"
        confidence = 0.0

    # Animated confidence ring with unique ID
    unique_id = uuid.uuid4().hex[:8]
    color = "#4CAF50" if confidence >= 80 else "#FF9800" if confidence >= 65 else "#F44336"
    ring_html = f'''
    <style>
    @keyframes pulse-{unique_id} {{
        0% {{ transform: scale(0.8); opacity: 0.5; }}
        100% {{ transform: scale(1); opacity: 1; }}
    }}
    </style>
    <div style="width:100px;height:100px;border-radius:50%;border:8px solid {color};
        display:flex;align-items:center;justify-content:center;
        font-size:16px;margin:auto;margin-top:10px;
        animation: pulse-{unique_id} 0.8s ease-in-out forwards;">
        {int(confidence)}
    </div>
    '''

    return prediction, ring_html

# Clear function
def clear_output():
    return "", ""

# Gradio UI
with gr.Blocks() as demo:

    # Title + Slogan + Instructions (left-aligned block with larger Labelo)
    gr.HTML("""
    <div style='text-align:left; margin-bottom: 1rem;'>
      <h1 style='margin-bottom: 4px; font-size: 32px; font-weight: 700;'>Labelo</h1>
      <div style='font-size:16px; color:white; font-style: italic; font-weight:500; margin-bottom: 8px;'>Upload. Recognize. Labelo.</div>
      <div style="font-size:13px; color:#888;">
        ℹ️ <b>Labelo</b> can recognize: <b>Cars</b>, <b>Plane</b>, and <b>Virat</b>.<br>
        📂 Supported formats: <code>.jpg</code>, <code>.png</code>, <code>.bmp</code>, <code>.tiff</code>, <code>.webp</code>, <code>.gif</code>
      </div>
    </div>
    """)

    gr.Markdown("### 🔍 Recognize Images")

    with gr.Row():
        recog_img = gr.Image(
            label="Upload Image",
            type="pil",
            image_mode="RGB",
            interactive=True,
            height=300,
            show_label=False,
            show_download_button=False,
            container=True
        )

    recog_btn = gr.Button("🔍 Recognize")
    recog_output = gr.Textbox(label="Prediction")
    confidence_ring = gr.HTML()

    recog_btn.click(fn=recognize, inputs=[recog_img], outputs=[recog_output, confidence_ring])
    recog_img.clear(fn=clear_output, outputs=[recog_output, confidence_ring])

demo.launch()
# To run this code write pipenv run python3 Labelo.py
