import os
import time
import requests
from PIL import Image
from io import BytesIO

# Hugging Face secret from GitHub Codespaces
HF_TOKEN = os.getenv("HF_TOKEN")

# Pre-trained image classification model
API_URL = "https://router.huggingface.co/hf-inference/models/google/vit-base-patch16-224"

IMAGE_PATH = "images/test.jpg"


def classify_image():

    if not HF_TOKEN:
        print("ERROR: HF_TOKEN secret not found.")
        return

    try:
        # Open image
        image = Image.open(IMAGE_PATH).convert("RGB")

        image_data = BytesIO()
        image.save(image_data, format="JPEG")

        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "image/jpeg"
        }

        print("Uploading image to AI model...")

        # Send image to Hugging Face
        for attempt in range(3):
            response = requests.post(
                API_URL,
                headers=headers,
                data=image_data.getvalue(),
                timeout=90
            )

            if response.status_code == 503 and attempt < 2:
                print("Model loading. Retrying...")
                time.sleep(10)
                continue

            break

        if response.status_code != 200:
            print("API Error:", response.status_code)
            print(response.text)
            return

        results = response.json()

        if not isinstance(results, list):
            print("Unexpected API response:", results)
            return

        print("\n===== IMAGE CLASSIFICATION RESULT =====")

        for item in results[:5]:
            label = item["label"]
            confidence = item["score"] * 100
            print(f"{label}: {confidence:.2f}%")

        print("=======================================")

    except FileNotFoundError:
        print("ERROR: Add your image at images/test.jpg")

    except Exception as error:
        print("ERROR:", error)


if __name__ == "__main__":
    classify_image()

