const imageInput = document.getElementById("imageInput");
const previewContainer = document.getElementById("preview-container");
const previewImage = document.getElementById("preview-image");

const predictButton = document.getElementById("predict-btn");

const resultSection = document.getElementById("result-section");

const predictedClass =
    document.getElementById("predicted-class");

const confidence =
    document.getElementById("confidence");

const originalImage =
    document.getElementById("original-image");

const heatmapImage =
    document.getElementById("heatmap-image");

const overlayImage =
    document.getElementById("overlay-image");

const loading =
    document.getElementById("loading");

const errorMessage =
    document.getElementById("error-message");


let selectedFile = null;


/* =========================
   IMAGE SELECTION
========================= */

imageInput.addEventListener("change", function () {

    const file = this.files[0];

    if (!file) {
        return;
    }

    selectedFile = file;

    const reader = new FileReader();

    reader.onload = function (event) {

        previewImage.src =
            event.target.result;

        previewContainer.style.display =
            "block";

        predictButton.disabled =
            false;

        resultSection.style.display =
            "none";

        errorMessage.style.display =
            "none";
    };

    reader.readAsDataURL(file);
});


/* =========================
   PREDICT DISEASE
========================= */

predictButton.addEventListener("click", async function () {

    if (!selectedFile) {
        return;
    }

    loading.style.display =
        "block";

    errorMessage.style.display =
        "none";

    predictButton.disabled =
        true;

    const formData = new FormData();

    formData.append(
        "image",
        selectedFile
    );


    try {

        const response = await fetch(
            "http://127.0.0.1:5000/predict",
            {
                method: "POST",
                body: formData
            }
        );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.error ||
                "Prediction failed."
            );
        }


        /* =========================
           DISPLAY RESULT
        ========================= */

        predictedClass.textContent =
            formatDiseaseName(
                data.predicted_class
            );


        confidence.textContent =
            `${data.confidence.toFixed(2)}%`;


        /* =========================
           DISPLAY IMAGES
        ========================= */

        originalImage.src =
            "data:image/jpeg;base64," +
            data.original_image;


        heatmapImage.src =
            "data:image/jpeg;base64," +
            data.gradcam_heatmap;


        overlayImage.src =
            "data:image/jpeg;base64," +
            data.gradcam_overlay;


        resultSection.style.display =
            "block";


        /* Scroll to result */

        resultSection.scrollIntoView({
            behavior: "smooth"
        });

    }


    catch (error) {

        console.error(error);

        errorMessage.textContent =
            "Unable to connect to the prediction server. " +
            "Please make sure the Flask backend is running.";

        errorMessage.style.display =
            "block";
    }


    finally {

        loading.style.display =
            "none";

        predictButton.disabled =
            false;
    }

});


/* =========================
   FORMAT DISEASE NAME
========================= */

function formatDiseaseName(name) {

    return name
        .replaceAll("_", " ")
        .replace(
            "Pepper bell",
            "Pepper Bell"
        )
        .replace(
            "Potato",
            "Potato"
        )
        .replace(
            "Tomato",
            "Tomato"
        );
}