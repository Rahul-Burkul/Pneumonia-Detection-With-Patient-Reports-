import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# =======================
# 1. Define FiLM & DenseNetFiLM
# =======================
from torchvision.models import densenet121, DenseNet121_Weights

class FiLMLayer(nn.Module):
    def __init__(self, n_features, n_meta):
        super().__init__()
        self.fc_gamma = nn.Linear(n_meta, n_features)
        self.fc_beta = nn.Linear(n_meta, n_features)

    def forward(self, x, meta):
        gamma = self.fc_gamma(meta).unsqueeze(2).unsqueeze(3)
        beta = self.fc_beta(meta).unsqueeze(2).unsqueeze(3)
        return gamma * x + beta

class DenseNetFiLM(nn.Module):
    def __init__(self, n_meta=4):
        super().__init__()
        backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.block0 = nn.Sequential(backbone.features.conv0,
                                    backbone.features.norm0,
                                    backbone.features.relu0,
                                    backbone.features.pool0)
        self.block1 = backbone.features.denseblock1
        self.trans1 = backbone.features.transition1
        self.block2 = backbone.features.denseblock2
        self.trans2 = backbone.features.transition2
        self.block3 = backbone.features.denseblock3
        self.trans3 = backbone.features.transition3
        self.block4 = backbone.features.denseblock4
        self.norm5 = backbone.features.norm5
        self.film = FiLMLayer(512, n_meta)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x, meta):
        out = self.block0(x)
        out = self.block1(out)
        out = self.trans1(out)
        out = self.block2(out)
        out = self.film(out, meta)
        out = self.trans2(out)
        out = self.block3(out)
        out = self.trans3(out)
        out = self.block4(out)
        out = self.norm5(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

# =======================
# 2. Load model
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenseNetFiLM(n_meta=4).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device), strict=False)
model.eval()

# =======================
# 3. Preprocessing transform
# =======================
input_size = 224
valid_tf = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =======================
# 4. Streamlit UI
# =======================
st.title("Pneumonia Detection with Grad-CAM")

# Upload X-ray image
uploaded_file = st.file_uploader("Upload Chest X-ray", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    # Metadata inputs
    # =======================
    st.sidebar.header("Patient Metadata")
    
    # Age input as text
    age_input = st.sidebar.text_input("Age", "50")  # default 50
    try:
        age = float(age_input)
        if age < 0 or age > 120:
            st.sidebar.warning("Age must be between 0 and 120")
            age = 50
    except:
        st.sidebar.warning("Invalid age input. Using default 50")
        age = 50
    
    # Sex
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    
    # Frontal / Lateral
    view_orientation = st.sidebar.selectbox("View Orientation", ["Frontal", "Lateral"])
    
    # AP / PA
    beam_direction = st.sidebar.selectbox("Beam Direction", ["PA", "AP"])
    
    # Encode metadata
    sex_binary = 0 if sex == "Male" else 1
    frontal_binary = 1 if view_orientation == "Frontal" else 0
    ap_binary = 1 if beam_direction == "AP" else 0
    
    # Create metadata tensor
    meta = torch.tensor([age/100, sex_binary, frontal_binary, ap_binary], dtype=torch.float32).unsqueeze(0).to(device)


    # Predict button
    if st.button("Predict"):
        # Encode metadata
        sex_binary = 0 if sex=="Male" else 1
        # Frontal / Lateral
        frontal_binary = 1 if view_orientation == "Frontal" else 0
        
        # AP / PA
        ap_binary = 1 if beam_direction == "AP" else 0

        meta = torch.tensor([age/100, sex_binary, frontal_binary, ap_binary], dtype=torch.float32).unsqueeze(0).to(device)

        # Preprocess image
        img_tensor = valid_tf(image).unsqueeze(0).to(device)

        # Grad-CAM wrapper
        class ModelWithMetaWrapper(nn.Module):
            def __init__(self, model, meta):
                super().__init__()
                self.model = model
                self.meta = meta
            def forward(self, x):
                return self.model(x, self.meta)

        wrapped_model = ModelWithMetaWrapper(model, meta)

        # Prediction + Grad-CAM
        with torch.no_grad():
            logits = model(img_tensor, meta).squeeze()
            prob = torch.sigmoid(logits).item()
            predicted_class = 1 if prob > 0.5 else 0

        targets = [BinaryClassifierOutputTarget(predicted_class)]
        cam = GradCAM(model=wrapped_model, target_layers=[model.norm5])
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0, :]

        # Resize original image to match CAM
        rgb_img = np.array(image.resize((input_size, input_size)))/255.0

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        st.image(cam_image, caption=f"Grad-CAM Overlay\nPredicted: {'Pneumonia' if predicted_class==1 else 'Normal'} ({prob:.2f})", use_container_width=True)
