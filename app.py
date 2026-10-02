import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class_names = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]


vgg = models.vgg16(weights=None)

vgg.classifier = nn.Sequential(
    nn.Linear(25088, 2025),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),

    nn.Linear(2025, 2025),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),

    nn.Linear(2025, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),

    nn.Linear(256, 10)
)



from pathlib import Path

# Find the folder where app.py is located
BASE_DIR = Path(__file__).resolve().parent

# Find the model file in the same folder
MODEL_PATH = BASE_DIR / "vgg16_eurosat.pth"

# Load trained weights
vgg.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

vgg.to(device)
vgg.eval()







transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])






st.title("🌍 EuroSAT Land Cover Classification")

st.write(
    "Upload a satellite image and the VGG16 model "
    "will predict its land-cover class."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict"):

        # Preprocesssing
        input_image = transform(image)


        input_image = input_image.unsqueeze(0)

        # Move to CPU/GPU
        input_image = input_image.to(device)

        # Prediction
        with torch.no_grad():
            output = vgg(input_image)

            probabilities = torch.softmax(
                output,
                dim=1
            )

            predicted_class = torch.argmax(
                probabilities,
                dim=1
            ).item()

        class_name = class_names[predicted_class]
        confidence = probabilities[0][predicted_class].item()

        st.success(
            f"Prediction: **{class_name}**"
        )

        st.write(
            f"Confidence: **{confidence * 100:.2f}%**"
        )