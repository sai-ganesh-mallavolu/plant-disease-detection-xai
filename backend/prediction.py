import os
import json
import numpy as np
import tensorflow as tf


IMG_SIZE = (224, 224)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "plant_disease_mobilenetv2.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "class_names.json"
)


# ==========================================
# Load trained model
# ==========================================

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ==========================================
# Load class names
# ==========================================

with open(
    CLASS_NAMES_PATH,
    "r"
) as file:

    class_names = json.load(file)


# ==========================================
# Prediction
# ==========================================

def predict_image(image_path):

    img = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    img_array = tf.keras.utils.img_to_array(
        img
    )

    img_batch = tf.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(
        img_batch,
        verbose=0
    )

    predicted_index = int(
        np.argmax(
            prediction[0]
        )
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = float(
        prediction[0][predicted_index] * 100
    )

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "predicted_index": predicted_index
    }