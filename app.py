from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import StreamingResponse
from PIL import Image, ImageOps, ImageDraw
from ultralytics import YOLO
from os import path, getcwd, environ, remove
from uuid import uuid4
import numpy as np
import cv2
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from remove_background import remove_background
import io

load_dotenv()

app = FastAPI()

images_dir = path.join(getcwd(), 'saved_images')
frame_image_path = "assets/HorizontalLogoTourtel.png"

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


@app.post("/merge-images/")
async def merge_images(files: list[UploadFile] = File(...)):
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

        # Return the URL for fetching the merged image
        return {"message": "Images merged successfully", "image_url": f"{merged_file_name}"}

    finally:
        # Remove all temporary files
        for temp_file in temp_files:
            if path.exists(temp_file):
                remove(temp_file)


@app.get("/image/{image_file}")
async def get_merged_image(image_file: str, request: Request):
    image_path = path.join(images_dir, image_file)
    if not path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    original_image = Image.open(image_path)
    frame_image = Image.open(frame_image_path)

    # Define padding and new dimensions
    padding = 10
    border = 10
    new_width = original_image.width // 2
    new_height = original_image.height + 2 * padding

    # Create a new image with padding and background color
    combined_image = Image.new("RGB", (new_width + 2 * border, new_height + 2 * border), "white")

    # Calculate cropping area for the original image
    left = (original_image.width - new_width) // 2
    right = left + new_width
    cropped_image = original_image.crop((left, 0, right, original_image.height * 0.8))

    # Create a new canvas for the cropped image with a white border
    bordered_image = Image.new("RGB", (new_width + 2 * border, original_image.height + 2 * border), "white")
    bordered_image.paste(cropped_image, (border, border))

    # Paste the bordered cropped image onto the combined image
    combined_image.paste(bordered_image, (0, padding))

    # Resize frame image to be smaller and calculate the position at the bottom
    frame_scale_factor = 0.42  # Adjust this value to control the size of the frame image
    frame_image = frame_image.resize((new_width + border, int((original_image.height * frame_scale_factor) / 2)))

    # Calculate position to paste the frame image at the bottom
    frame_x = 0
    frame_y = new_height + 2 * border - frame_image.height

    # Paste frame image on the combined image
    combined_image.paste(frame_image, (frame_x, frame_y), frame_image)

    # Convert combined image to bytes
    image_bytes = io.BytesIO()
    combined_image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    return StreamingResponse(image_bytes, media_type="image/jpeg")
