from flask import Flask, request, jsonify
from flask_cors import CORS

import os
import time

from prediction import predict_image


app = Flask(__name__)

CORS(app)


@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Plant Disease Detection API is running!"
    })


@app.route("/predict-test", methods=["POST"])
def predict_test():

    start_time = time.time()

    print("========== PREDICTION TEST START ==========")

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
        "temp_test_image"
    )

    try:

        print("1. Saving image...")

        image.save(temp_path)

        print(
            "2. Image saved in:",
            f"{time.time() - start_time:.2f}s"
        )

        print("3. Starting TensorFlow prediction...")

        prediction_start = time.time()

        result = predict_image(
            temp_path
        )

        prediction_time = (
            time.time() - prediction_start
        )

        print(
            "4. Prediction completed in:",
            f"{prediction_time:.2f}s"
        )

        print(
            "Predicted class:",
            result["predicted_class"]
        )

        print(
            "Confidence:",
            result["confidence"]
        )

        print("========== PREDICTION TEST END ==========")

        return jsonify({

            "status": "success",

            "predicted_class":
                result["predicted_class"],

            "confidence":
                result["confidence"],

            "prediction_time":
                prediction_time

        })

    except Exception as e:

        print(
            "Prediction test error:",
            str(e)
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