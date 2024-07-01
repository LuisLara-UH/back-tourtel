from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
import numpy as np
import cv2
from io import BytesIO

app = FastAPI()

# Load your YOLO model (adjust the model path as necessary)
model = YOLO('yolov8s.pt')  # Replace with your actual model file


@app.post("/merge-images/")
async def merge_images(files: list[UploadFile] = File(...)):
    base_pil_image = None

    for i, file in enumerate(files):
        # Read the uploaded file
        contents = await file.read()
        np_image = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        # Use the first image as the base
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

                    # Convert ROI to PIL Image format for easy pasting
                    person_pil = Image.fromarray(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))

                    # Create a mask for pasting
                    mask = ImageOps.invert(person_pil.convert('L'))

                    # Paste the extracted person onto the base image
                    base_pil_image.paste(person_pil, (x1, y1), mask)

    # Save the merged image into a BytesIO object
    merged_image_io = BytesIO()
    base_pil_image.save(merged_image_io, format='JPEG')
    merged_image_io.seek(0)

    return StreamingResponse(merged_image_io, media_type="image/jpeg")
