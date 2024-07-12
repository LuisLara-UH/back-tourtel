from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import StreamingResponse
from PIL import Image, ImageOps, ImageChops
from ultralytics import YOLO
from os import path, getcwd, environ, remove
from uuid import uuid4

import numpy as np
import cv2
import requests
import json
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

images_dir = path.join(getcwd(), 'saved_images')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO('yolov8s.pt')

API_KEY = environ.get('API_KEY')
SERVER_URL = environ.get('SERVER_URL')


def get_segmentation_mask(image_url):
    url = "https://modelslab.com/api/v6/image_editing/removebg_mask"
    payload = json.dumps({
        "key": API_KEY,
        "image": image_url,
        "post_process_mask": False,
        "only_mask": True,
        "alpha_matting": False
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    if response.status_code == 200:
        mask_url = response.json().get('output_url')
        if mask_url:
            mask_response = requests.get(mask_url)
            if mask_response.status_code == 200:
                return Image.open(BytesIO(mask_response.content))
    return None


@app.post("/merge-images/")
async def merge_images(files: list[UploadFile] = File(...)):
    base_pil_image = None

    for i, file in enumerate(files):
        # Read the uploaded file
        contents = await file.read()
        np_image = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        # Save the image temporarily to get a URL for the API request
        temp_image_path = path.join(images_dir, f"temp_{uuid4().hex}.jpg")
        cv2.imwrite(temp_image_path, image)

        if i == 0:
            base_pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            continue

        # Detect people in the current image
        results = model([image])

        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.cls == 0:  # Assuming class '0' represents 'person'
                    # Extract the bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()  # Convert tensor to list

                    # Ensure coordinates are integers
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    # Extract the person from the image
                    person_roi = image[y1:y2, x1:x2]

                    # Convert ROI to PIL Image format for processing
                    person_pil = Image.fromarray(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))

                    # Save the person image temporarily to get a URL for the API request
                    person_image_path = path.join(images_dir, f"person_{uuid4().hex}.jpg")
                    person_pil.save(person_image_path)
                    person_image_url = f"{SERVER_URL}/image/{path.basename(person_image_path)}"

                    # Get the segmentation mask from the API
                    mask_pil = get_segmentation_mask(person_image_url)

                    if mask_pil:
                        # Extract the segmented person using the mask
                        person_segmented_pil = ImageChops.multiply(person_pil, mask_pil.convert('RGB'))

                        # Paste the segmented person onto the base image with the mask
                        base_pil_image.paste(person_segmented_pil, (x1, y1), mask_pil)

                    try:
                        remove(person_image_path)
                    except OSError as e:
                        print(f"Error: {person_image_path} : {e.strerror}")

        # Generate a unique filename for the merged image
        merged_file_name = f"{uuid4().hex}.jpg"
        merged_image_path = path.join(images_dir, merged_file_name)

        # Save the merged image to the disk
        base_pil_image.save(merged_image_path, format='JPEG')

        # Return the URL for fetching the merged image
        return {"message": "Images merged successfully", "image_url": f"{SERVER_URL}/image/{merged_file_name}"}


@app.get("/image/{image_file}")
async def get_merged_image(image_file: str, request: Request):
    image_path = path.join(images_dir, image_file)
    if not path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    image_file = open(image_path, "rb")
    return StreamingResponse(image_file, media_type="image/jpeg")
