# 🌿 Plant Disease Detection & Explainable AI

An AI-powered Plant Disease Detection system that identifies plant diseases from leaf images using MobileNetV2 and explains the model's prediction using Grad-CAM (Gradient-weighted Class Activation Mapping).

The project includes a trained deep learning model, a Flask backend API, and a simple web frontend for uploading leaf images and viewing the prediction with visual explanations.

---

## 📌 Project Overview

Plant diseases can affect crop quality and agricultural productivity. Identifying diseases manually from leaf symptoms can be difficult, especially when different diseases have similar visual characteristics.

This project uses Deep Learning and Explainable AI (XAI) to detect plant diseases from leaf images.

The system:

1. Accepts a plant leaf image from the user.
2. Processes the image using the trained MobileNetV2-based model.
3. Predicts the most likely plant disease.
4. Displays the prediction confidence.
5. Generates a Grad-CAM heatmap.
6. Creates an overlay showing the regions that influenced the model's prediction.

The main goal is not only to make a prediction, but also to provide a visual explanation of where the model focused while making that prediction.

---

## 🎯 Key Features

- 🌱 Plant leaf image upload
- 🤖 Deep learning-based disease classification
- 🧠 MobileNetV2 transfer learning
- 📊 Prediction confidence
- 🔥 Grad-CAM heatmap
- 🖼️ Grad-CAM overlay visualization
- 🌐 Flask REST API
- 💻 Simple web frontend
- 📱 Responsive frontend design
- 🔗 Separate frontend and backend architecture
- 🐙 GitHub-ready project structure

---

## 🏗️ Project Architecture

```text
                    ┌──────────────────────┐
                    │      Frontend        │
                    │  HTML + CSS + JS     │
                    └──────────┬───────────┘
                               │
                               │ HTTP POST
                               │ /predict
                               ▼
                    ┌──────────────────────┐
                    │    Flask Backend     │
                    │       app.py         │
                    └──────────┬───────────┘
                               │
                  ┌────────────┴────────────┐
                  │                         │
                  ▼                         ▼
        ┌──────────────────┐      ┌──────────────────┐
        │   Prediction     │      │     Grad-CAM     │
        │   prediction.py  │      │    gradcam.py    │
        └────────┬─────────┘      └────────┬─────────┘
                 │                         │
                 ▼                         ▼
        ┌──────────────────┐      ┌──────────────────┐
        │ MobileNetV2      │      │ Feature Maps +   │
        │ Trained Model    │      │ Gradients        │
        └──────────────────┘      └────────┬─────────┘
                                           │
                                           ▼
                                  ┌──────────────────┐
                                  │ Heatmap + Overlay │
                                  └──────────────────┘
```

---

## 🧠 Model

The project uses MobileNetV2 with transfer learning.

The MobileNetV2 base model is loaded with ImageNet weights and its original classification head is replaced with a custom classification head for the plant disease dataset.

### Model Architecture

```text
Input Image
    │
    ▼
224 × 224 × 3
    │
    ▼
Data Augmentation
    │
    ├── Random Flip
    ├── Random Rotation
    ├── Random Zoom
    ├── Random Translation
    └── Random Contrast
    │
    ▼
MobileNetV2
    │
    ▼
Global Average Pooling
    │
    ▼
Dropout (0.30)
    │
    ▼
Dense Layer (128, ReLU)
    │
    ▼
Dropout (0.20)
    │
    ▼
Softmax Output
    │
    ▼
15 Plant Classes
```

---

## 🔧 Training Configuration

The training configuration follows the original project notebook.

| Parameter | Value |
|---|---|
| Image Size | `224 × 224` |
| Batch Size | `32` |
| Validation Split | `0.2` |
| Random Seed | `42` |
| Optimizer | Adam |
| Learning Rate | `0.0001` |
| Loss Function | Sparse Categorical Crossentropy |
| Maximum Epochs | `10` |
| Early Stopping Patience | `3` |
| Base Model | MobileNetV2 |
| Pretrained Weights | ImageNet |

### Regularization

The classification head uses:

- Dropout `0.30`
- Dense layer with L2 regularization `0.001`
- Dropout `0.20`

---

## 🌱 Dataset

The dataset contains 20,638 images belonging to 15 plant disease/health classes.

The dataset was divided into:

- Training: 16,511 images
- Validation: 4,127 images

### Classes

The model can classify the following 15 classes:

```text
1. Pepper__bell___Bacterial_spot
2. Pepper__bell___healthy
3. Potato___Early_blight
4. Potato___Late_blight
5. Potato___healthy
6. Tomato_Bacterial_spot
7. Tomato_Early_blight
8. Tomato_Late_blight
9. Tomato_Leaf_Mold
10. Tomato_Septoria_leaf_spot
11. Tomato_Spider_mites_Two_spotted_spider_mite
12. Tomato__Target_Spot
13. Tomato__Tomato_YellowLeaf__Curl_Virus
14. Tomato__Tomato_mosaic_virus
15. Tomato_healthy
```

---

## 📊 Model Performance

The trained model achieved approximately 86% validation accuracy on the validation dataset.

The classification report showed:

```text
Accuracy: 0.86

Macro Average:
Precision: 0.87
Recall:    0.83
F1-Score: 0.84

Weighted Average:
Precision: 0.87
Recall:    0.86
F1-Score: 0.86
```

Performance varies across individual classes, which is expected because some disease categories contain fewer images or have visually similar symptoms.

---

## 🔥 Explainable AI — Grad-CAM

A major part of this project is Explainable AI.

Instead of displaying only the predicted disease, the system also shows the regions of the leaf that contributed to the model's prediction.

The project uses Grad-CAM.

### Grad-CAM Workflow

```text
Input Leaf Image
       │
       ▼
MobileNetV2
       │
       ▼
Feature Maps
       │
       ▼
Calculate Gradients
       │
       ▼
Global Average of Gradients
       │
       ▼
Weighted Feature Maps
       │
       ▼
ReLU
       │
       ▼
Normalize
       │
       ▼
Grad-CAM Heatmap
       │
       ▼
Overlay with Original Image
```

The final application displays three images:

### 1. Original Image

The original leaf image uploaded by the user.

### 2. Grad-CAM Heatmap

A color-coded visualization showing the regions that contributed to the model's prediction.

### 3. Grad-CAM Overlay

The heatmap is placed over the original leaf image to make the explanation easier to understand.

---

## 📁 Project Structure

```text
Plant_Disease_Detection/
│
├── .venv/
│
├── dataset/
│   └── Plant disease image classes
│
├── notebook/
│   └── Plant_Disease_Detection.ipynb
│
├── backend/
│   │
│   ├── app.py
│   ├── prediction.py
│   ├── gradcam.py
│   ├── class_names.json
│   ├── requirements.txt
│   │
│   └── model/
│       └── plant_disease_mobilenetv2.keras
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── .gitignore
│
└── README.md
```

---

## 📂 Backend Files

### `app.py`

The main Flask application.

It:

- Starts the Flask server
- Handles API requests
- Receives uploaded images
- Calls the prediction function
- Generates Grad-CAM
- Converts generated images to Base64
- Returns the prediction and explanation to the frontend

### `prediction.py`

Responsible for loading the trained model and performing disease prediction.

It:

- Loads the `.keras` model
- Loads the class names
- Resizes the image to `224 × 224`
- Runs model prediction
- Finds the predicted class
- Calculates confidence

### `gradcam.py`

Contains the Grad-CAM implementation.

It:

- Creates the Grad-CAM model
- Extracts MobileNetV2 feature maps
- Calculates gradients
- Generates the heatmap
- Creates the colored heatmap
- Creates the overlay image

### `class_names.json`

Contains the 15 class names used by the trained model.

### `model/plant_disease_mobilenetv2.keras`

The trained MobileNetV2-based plant disease classification model.

---

## 🎨 Frontend

The frontend is built using:

- HTML
- CSS
- JavaScript

The interface allows the user to:

1. Select a leaf image.
2. Preview the selected image.
3. Click **Predict Disease**.
4. Send the image to the Flask API.
5. View the predicted disease.
6. View confidence.
7. View the Grad-CAM heatmap.
8. View the Grad-CAM overlay.

---

## 🔌 API

The backend exposes a prediction endpoint:

```text
POST /predict
```

### Request

The request should contain the image as a multipart form-data field named:

```text
image
```

Example:

```bash
curl -X POST   -F "image=@leaf.jpg"   http://127.0.0.1:5000/predict
```

### Response

The API returns information similar to:

```json
{
    "predicted_class": "Tomato_healthy",
    "confidence": 99.87,
    "original_image": "...",
    "gradcam_heatmap": "...",
    "gradcam_overlay": "..."
}
```

The three image fields are returned as Base64-encoded JPEG data so that the frontend can directly display them.

---

# 🚀 How to Run the Project Locally

## 1. Clone the Repository

```bash
git clone https://github.com/sai-ganesh-mallavolu/plant-disease-detection-xai.git
```

Move into the project:

```bash
cd plant-disease-detection-xai
```

---

## 2. Create Virtual Environment

Create the virtual environment in the project root:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scriptsctivate
```

---

## 3. Install Backend Dependencies

Move to the backend folder:

```powershell
cd backend
```

Install the required packages:

```powershell
pip install -r requirements.txt
```

---

## 4. Start the Flask Backend

Run:

```powershell
python app.py
```

The backend will start at:

```text
http://127.0.0.1:5000
```

You should see:

```text
Creating Grad-CAM components...
Grad-CAM components created successfully!
```

---

## 5. Run the Frontend

Open the `frontend` folder.

You can open:

```text
frontend/index.html
```

in a browser.

For a better local development experience, you can also use a local development server such as VS Code Live Server.

The frontend sends prediction requests to:

```text
http://127.0.0.1:5000/predict
```

---

# 🔄 Complete Application Workflow

```text
User
 │
 │ Upload Leaf Image
 ▼
Frontend
 │
 │ POST /predict
 ▼
Flask Backend
 │
 ├───────────────┐
 │               │
 ▼               ▼
Prediction     Grad-CAM
 │               │
 ▼               ▼
Disease        Heatmap
 │               │
 └───────┬───────┘
         │
         ▼
      Response
         │
         ▼
     Frontend
         │
         ├── Predicted Disease
         ├── Confidence
         ├── Original Image
         ├── Heatmap
         └── Overlay
```

---

## 🧪 Testing

The backend was tested using the Flask API.

A successful prediction response contains:

```text
Predicted Class
Confidence
Original Image
Grad-CAM Heatmap
Grad-CAM Overlay
```

The frontend was also tested by uploading leaf images and displaying the prediction together with the Grad-CAM visual explanation.

---

## 🛠️ Technologies Used

### Programming Languages

- Python
- JavaScript
- HTML
- CSS

### Machine Learning / Deep Learning

- TensorFlow
- Keras
- MobileNetV2
- NumPy
- OpenCV
- Pillow

### Explainable AI

- Grad-CAM

### Backend

- Flask
- Flask-CORS

### Development

- Jupyter Notebook
- VS Code
- Git
- GitHub

---

## 📓 Notebook

The complete machine learning workflow is documented in:

```text
notebook/Plant_Disease_Detection.ipynb
```

The notebook contains the major stages of the project, including:

- Dataset loading
- Dataset inspection
- Class analysis
- Data augmentation
- MobileNetV2 model creation
- Model training
- Model evaluation
- Classification report
- Confusion matrix
- Single-image prediction
- Grad-CAM generation
- Grad-CAM visualization

The trained model is then used by the separate Flask backend for runtime prediction.

---

## 💡 Why Grad-CAM?

A classification model normally gives an output such as:

```text
Tomato Late Blight
Confidence: 99%
```

But the prediction alone does not show why the model made that prediction.

Grad-CAM provides an additional visual explanation by highlighting the areas of the image that contributed to the prediction.

This makes the project more interpretable than a simple image classification system.

---

## ⚠️ Important Note

This project is intended as a machine learning and Explainable AI project.

The model's predictions should not be treated as a replacement for professional agricultural diagnosis or expert advice.

The reported validation performance is based on the project's validation dataset and may not represent performance on every real-world plant image, camera condition, lighting condition, or disease stage.

---

## 🔮 Future Improvements

Possible future improvements include:

- 📱 Mobile-friendly application
- ☁️ Cloud deployment
- 🌱 Support for additional plant species
- 📈 Improved model performance
- 📷 Real-time camera prediction
- 🗃️ Prediction history
- 👨‍🌾 Farmer-friendly disease information
- 💊 Suggested treatment information from reliable agricultural sources
- 🔐 User authentication
- 📊 Prediction analytics dashboard

---

## 🎯 Project Goal

The main goal of this project is to combine:

**Deep Learning + Computer Vision + Explainable AI + Web Development**

into a complete application that can:

> **Detect a possible plant disease from a leaf image and visually explain the regions that influenced the model's prediction.**

---

## 👨‍💻 Author

**Mallavolu Sai Ganesh**

📧 **Email:** mallavolusaiganesh@gmail.com

🔗 **GitHub:**  
https://github.com/sai-ganesh-mallavolu

---

⭐ **If you find this project useful, consider giving the repository a star!**
