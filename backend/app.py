from flask import Flask, request, jsonify
from flask_cors import CORS

import os
import base64
import cv2
import tensorflow as tf

from prediction import predict_image, model
from gradcam import (
    create_gradcam_components,
    generate_gradcam,
    create_gradcam_overlay
)


# ==========================================
# Flask Application
# ==========================================

app = Flask(__name__)

CORS(
    app,
    resources={
        r"/*": {
            "origins": [
                "https://plant-disease-detection-xai.vercel.app"
            ]
        }
    }
)


# ==========================================
# Grad-CAM Components
# ==========================================

print("Creating Grad-CAM components...")

(
    gradcam_base,
    gap_layer,
    dense_layer,
    final_layer
) = create_gradcam_components(model)

print("Grad-CAM components created successfully!")


# ==========================================
# Home Route
# ==========================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Plant Disease Detection API is running!"
    })


# ==========================================
# Prediction + Grad-CAM API
# ==========================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    if "image" not in request.files:

        return jsonify({
            "error": "No image uploaded."
        }), 400

    image = request.files["image"]

    if image.filename == "":

        return jsonify({
            "error": "No image selected."
        }), 400

    temp_path = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "temp_gradcam_image"
    )

    try:

        # ----------------------------------
        # Save uploaded image
        # ----------------------------------

        image.save(temp_path)


        # ----------------------------------
        # Prediction
        # ----------------------------------

        result = predict_image(
            temp_path
        )

        predicted_index = result[
            "predicted_index"
        ]


        # ----------------------------------
        # Prepare image for Grad-CAM
        # ----------------------------------

        img = tf.keras.utils.load_img(
            temp_path,
            target_size=(224, 224)
        )

        img_array = tf.keras.utils.img_to_array(
            img
        )

        img_batch = tf.expand_dims(
            img_array,
            axis=0
        )


        # ----------------------------------
        # Generate Grad-CAM
        # ----------------------------------

        heatmap = generate_gradcam(
            img_batch,
            predicted_index,
            gradcam_base,
            gap_layer,
            dense_layer,
            final_layer
        )


        # ----------------------------------
        # Create Grad-CAM overlay
        # ----------------------------------

        (
            original,
            heatmap_resized,
            superimposed
        ) = create_gradcam_overlay(
            temp_path,
            heatmap
        )


        # ----------------------------------
        # Convert images to Base64
        # ----------------------------------

        def image_to_base64(array):

            success, buffer = cv2.imencode(
                ".jpg",
                cv2.cvtColor(
                    array,
                    cv2.COLOR_RGB2BGR
                )
            )

            if not success:

                raise ValueError(
                    "Unable to encode image."
                )

            return base64.b64encode(
                buffer
            ).decode("utf-8")


        original_base64 = image_to_base64(
            original
        )

        heatmap_base64 = image_to_base64(
            heatmap_resized
        )

        overlay_base64 = image_to_base64(
            superimposed
        )


        # ----------------------------------
        # Final Response
        # ----------------------------------

        return jsonify({

            "predicted_class":
                result["predicted_class"],

            "confidence":
                result["confidence"],

            "original_image":
                original_base64,

            "gradcam_heatmap":
                heatmap_base64,

            "gradcam_overlay":
                overlay_base64
        })


    except Exception as e:

        print("Prediction error:", e)

        return jsonify({
            "error": str(e)
        }), 500


    finally:

        # Remove temporary file
        if os.path.exists(temp_path):

            os.remove(
                temp_path
            )


# ==========================================
# Run Application
# ==========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )