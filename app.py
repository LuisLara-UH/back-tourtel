from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from ultralytics import YOLO
import numpy as np
import cv2
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image, ImageOps, ImageChops
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
import uuid
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO('yolov8s.pt')  # Replace with your actual model file

segmentation_model = deeplabv3_resnet101(pretrained=True).eval()

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_dir = 'saved_images'
os.makedirs(image_dir, exist_ok=True)


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

                    # Convert ROI to PIL Image format for processing
                    person_pil = Image.fromarray(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))

                    # Preprocess the image for the segmentation model
                    input_tensor = preprocess(person_pil).unsqueeze(0)

                    # Perform the segmentation
                    with torch.no_grad():
                        output = segmentation_model(input_tensor)['out'][0]
                    output_predictions = output.argmax(0).byte().cpu().numpy()

                    # Create a mask for the person
                    person_mask = (output_predictions == 15).astype(np.uint8)  # Class '15' is 'person' in COCO dataset

                    # Convert the mask to a PIL Image
                    mask_pil = Image.fromarray(person_mask * 255, mode='L')

                    # Extract the segmented person using the mask
                    person_segmented_pil = ImageChops.multiply(person_pil, mask_pil.convert('RGB'))

                    # Paste the segmented person onto the base image with the mask
                    base_pil_image.paste(person_segmented_pil, (x1, y1), mask_pil)

    # Save the merged image to the filesystem
    image_id = str(uuid.uuid4())
    image_path = os.path.join(image_dir, f"{image_id}.jpg")
    base_pil_image.save(image_path, format='JPEG')

    return {"url": f"http://146.59.160.23:8000/image/{image_id}"}


@app.get("/image/{image_id}")
async def get_image(image_id: str):
    image_path = os.path.join(image_dir, f"{image_id}.jpg")
    if os.path.exists(image_path):
        return FileResponse(image_path, media_type='image/jpeg')
    return {"error": "Image not found"}
