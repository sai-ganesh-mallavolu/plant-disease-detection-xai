from flask import Flask, request, jsonify
from flask_cors import CORS

import os
import base64
import cv2
import tensorflow as tf
import time

from prediction import predict_image, model
from gradcam import (
    create_gradcam_components,
    generate_gradcam,
    create_gradcam_overlay
)


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


print("Creating Grad-CAM components...")


(
    gradcam_base,
    gap_layer,
    dense_layer,
    final_layer
) = create_gradcam_components(model)


print("Grad-CAM components created successfully!")


@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Plant Disease Detection API is running!"
    })


@app.route("/predict", methods=["POST"])
def predict():

    request_start = time.time()

    print("\n==============================")
    print("PREDICTION REQUEST STARTED")
    print("==============================")


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

        # ==============================
        # STEP 1 — SAVE IMAGE
        # ==============================

        start = time.time()

        image.save(temp_path)

        print(
            f"1. Image save time: "
            f"{time.time() - start:.2f} seconds"
        )


        # ==============================
        # STEP 2 — PREDICTION
        # ==============================

        start = time.time()

        result = predict_image(
            temp_path
        )

        print(
            f"2. Prediction time: "
            f"{time.time() - start:.2f} seconds"
        )


        predicted_index = result[
            "predicted_index"
        ]


        print(
            "Predicted class:",
            result["predicted_class"]
        )

        print(
            "Confidence:",
            result["confidence"]
        )


        # ==============================
        # STEP 3 — LOAD IMAGE
        # ==============================

        start = time.time()

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


        print(
            f"3. Image loading/preparation time: "
            f"{time.time() - start:.2f} seconds"
        )


        # ==============================
        # STEP 4 — GRAD-CAM
        # ==============================

        start = time.time()

        print("Starting Grad-CAM...")

        heatmap = generate_gradcam(
            img_batch,
            predicted_index,
            gradcam_base,
            gap_layer,
            dense_layer,
            final_layer
        )

        print(
            f"4. Grad-CAM time: "
            f"{time.time() - start:.2f} seconds"
        )


        # ==============================
        # STEP 5 — CREATE OVERLAY
        # ==============================

        start = time.time()

        (
            original,
            heatmap_resized,
            superimposed
        ) = create_gradcam_overlay(
            temp_path,
            heatmap
        )

        print(
            f"5. Overlay creation time: "
            f"{time.time() - start:.2f} seconds"
        )


        # ==============================
        # STEP 6 — BASE64 ENCODING
        # ==============================

        start = time.time()


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


        print(
            f"6. Base64 encoding time: "
            f"{time.time() - start:.2f} seconds"
        )


        # ==============================
        # STEP 7 — TOTAL TIME
        # ==============================

        total_time = (
            time.time() - request_start
        )


        print(
            f"7. TOTAL REQUEST TIME: "
            f"{total_time:.2f} seconds"
        )


        print("==============================")
        print("PREDICTION REQUEST COMPLETED")
        print("==============================\n")


        # ==============================
        # RESPONSE
        # ==============================

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

        print(
            "Prediction error:",
            str(e)
        )


        print(
            f"Request failed after: "
            f"{time.time() - request_start:.2f} seconds"
        )


        return jsonify({
            "error": str(e)
        }), 500


    finally:

        if os.path.exists(temp_path):

            os.remove(temp_path)


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )