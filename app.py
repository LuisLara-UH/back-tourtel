from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import StreamingResponse
from PIL import Image, ImageOps, ImageChops
from ultralytics import YOLO
from os import path, getcwd, environ, remove
from uuid import uuid4
import numpy as np
import cv2
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from remove_background import remove_background  # Import your function

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
        return {"message": "Images merged successfully", "image_url": f"{SERVER_URL}/image/{merged_file_name}"}

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

    image_file = open(image_path, "rb")
    return StreamingResponse(image_file, media_type="image/jpeg")
