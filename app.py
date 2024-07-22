from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form, Query
from fastapi.responses import StreamingResponse, HTMLResponse
from PIL import Image, ImageOps, ImageDraw
from ultralytics import YOLO
from os import path, getcwd, environ, remove, makedirs, listdir
from uuid import uuid4
import numpy as np
import cv2
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from remove_background import remove_background
import io
import json
from datetime import datetime, timezone

load_dotenv()

app = FastAPI()

images_dir = path.join(getcwd(), 'saved_images')
metadata_dir = path.join(getcwd(), 'metadata')
frame_image_path = "assets/tourtelLogoHD.png"

horizontal_displacement = 0

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

if not path.exists(images_dir):
    makedirs(images_dir)
if not path.exists(metadata_dir):
    makedirs(metadata_dir)


@app.post("/merge-images/")
async def merge_images(people_count: int = Query(...), files: list[UploadFile] = File(...)):
    base_pil_image = None
    temp_files = []

    try:
        for i, file in enumerate(files):
            # Read the uploaded file
            contents = await file.read()
            np_image = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

            # Save the image temporarily to get a URL for the API request
            temp_image_path = path.join(images_dir, f"temp_{uuid4().hex}.jpg")
            cv2.imwrite(temp_image_path, image)
            temp_files.append(temp_image_path)

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
                        temp_files.append(person_image_path)

                        # Remove the background using the external API
                        output_image_path = path.join(images_dir, f"person_segmented_{uuid4().hex}.png")
                        remove_background(person_image_path, output_image_path)
                        temp_files.append(output_image_path)

                        # Open the segmented image
                        segmented_person_pil = Image.open(output_image_path)

                        # Convert the image to RGBA format to use the alpha channel as the mask
                        segmented_person_pil = segmented_person_pil.convert("RGBA")
                        mask_pil = segmented_person_pil.split()[-1]  # Get the alpha channel as mask

                        # Paste the segmented person onto the base image with the mask
                        base_pil_image.paste(segmented_person_pil, (x1, y1), mask_pil)

        # Generate a unique filename for the merged image
        merged_file_name = f"{uuid4().hex}.jpg"
        merged_image_path = path.join(images_dir, merged_file_name)

        # Save the merged image to the disk
        base_pil_image.save(merged_image_path, format='JPEG')

        # Save metadata
        metadata = {
            "created_at": datetime.now(timezone.tzname('Europe/France')),
            "people_count": people_count
        }
        metadata_path = path.join(metadata_dir, f"{merged_file_name}.json")
        with open(metadata_path, 'w') as metadata_file:
            json.dump(metadata, metadata_file)

        # Return the URL for fetching the merged image
        return {"message": "Images merged successfully", "image_url": f"{merged_file_name}"}

    finally:
        # Remove all temporary files
        for temp_file in temp_files:
            if path.exists(temp_file):
                remove(temp_file)


@app.get("/crop-form/", response_class=HTMLResponse)
async def crop_form():
    html_content = """
    <html>
        <head>
            <title>Crop Image</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                }
                form {
                    background-color: #fff;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    width: 300px;
                    text-align: center;
                }
                label, input {
                    display: block;
                    width: 100%;
                    margin-bottom: 10px;
                }
                input[type="number"] {
                    padding: 8px;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                }
                input[type="submit"] {
                    background-color: #4CAF50;
                    color: white;
                    padding: 10px 15px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                }
                input[type="submit"]:hover {
                    background-color: #45a049;
                }
            </style>
        </head>
        <body>
            <form action="/crop-image/" method="post">
                <label for="x_position">X-axis Position (pixels):</label>
                <input type="number" id="x_position" name="x_position" value="0" required><br><br>
                <input type="submit" value="Set Cropped Image Position">
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/crop-image/")
async def crop_image(x_position: int = Form(0)):
    global horizontal_displacement
    horizontal_displacement = x_position
    return {"message": "Position value received", "x_position": x_position}


@app.get("/image/{image_file}")
async def get_merged_image(image_file: str, request: Request):
    global horizontal_displacement

    image_path = path.join(images_dir, image_file)
    if not path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    original_image = Image.open(image_path)
    frame_image = Image.open(frame_image_path)

    # Define resulting image dimensions
    result_width = 892
    result_height = 1206
    padding = 0
    border = 30

    # Calculate new dimensions for the cropped image to fit the height and maintain aspect ratio
    scale_factor = ((result_height - 200) - 2 * border) / original_image.height
    new_height = int(original_image.height * scale_factor)
    new_width = int(original_image.width * scale_factor)

    # Calculate cropping area for the original image
    left = (new_width - result_width + 2 * border) // 2 + horizontal_displacement
    right = left + result_width - 2 * border
    cropped_image = original_image.resize((new_width, new_height), Image.LANCZOS).crop((left, 0, right, new_height))

    # Create a new canvas for the cropped image with a white border
    bordered_image = Image.new("RGB", (result_width, new_height + 2 * border), "white")
    bordered_image.paste(cropped_image, (border, border))

    # Create the final combined image with padding
    combined_image = Image.new("RGB", (result_width, result_height), "white")
    combined_image.paste(bordered_image, (0, padding))

    # Resize frame image to fit within the bottom area
    frame_scale_factor = 0.3
    frame_width = int(frame_scale_factor * frame_image.width)
    frame_image = frame_image.resize(
        (
            int(frame_scale_factor * frame_image.width),
            int(frame_scale_factor * frame_image.height)
        ),
        Image.LANCZOS
    )

    # Calculate position to paste the frame image at the bottom
    frame_x = (result_width - frame_width) // 2
    frame_y = result_height - padding - frame_image.height

    # Paste frame image on the combined image
    combined_image.paste(frame_image, (frame_x, frame_y), frame_image)

    # Convert combined image to bytes
    image_bytes = io.BytesIO()
    combined_image.save(image_bytes, format="JPEG", quality=95)
    image_bytes.seek(0)

    return StreamingResponse(image_bytes, media_type="image/jpeg")


@app.get("/metadata/")
async def get_all_metadata():
    metadata = []
    files = listdir(metadata_dir)

    for file in files:
        if file.endswith('.json'):
            file_path = path.join(metadata_dir, file)
            with open(file_path, 'r') as metadata_file:
                metadata.append(json.load(metadata_file))

    return metadata
