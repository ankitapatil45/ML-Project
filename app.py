import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode

# -----------------------
# Load model config
# -----------------------
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

# Update for your dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8   # total classes
cfg.MODEL.WEIGHTS = "model_final.pth"  # path to trained weights
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

# 🔑 Force CPU mode
cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)

# -----------------------
# Define class names (must match your dataset order)
# -----------------------
class_names = [
    "Dent",
    "Scratch",
    "Broken part",
    "Paint chip",
    "Missing part",
    "Flaking",
    "Corrosion",
    "Cracked"
]

# -----------------------
# Streamlit UI
# -----------------------
st.title("🚗 Car Damage Detection App")

# Upload image
uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    # Run inference
    outputs = predictor(img_np)

    # -----------------------
    # Visualize predictions
    # -----------------------
    v = Visualizer(
        img_np[:, :, ::-1],
        metadata={"thing_classes": class_names},
        scale=0.8,
        instance_mode=ColorMode.IMAGE_BW  # keep background light
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Show output image with labels
    st.image(out.get_image()[:, :, ::-1], caption="Detected Damages", use_column_width=True)

    # Print all detections with confidence
    pred_classes = outputs["instances"].pred_classes.cpu().numpy()
    scores = outputs["instances"].scores.cpu().numpy()

    st.subheader("🔎 Detected Damages:")
    for i, cls in enumerate(pred_classes):
        st.write(f"- **{class_names[cls]}** (Confidence: {scores[i]:.2f})")
