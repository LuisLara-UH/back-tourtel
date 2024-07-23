# Environment Variables for Image Processing

This document explains the environment variables used in the FastAPI application for image processing. These variables can be configured in the `.env` file to adjust the dimensions and scaling factors of the images.

## Environment Variables

### RESULT_WIDTH

- **Description**: Defines the width of the resulting image.
- **Default Value**: 892
- **Usage**: `RESULT_WIDTH=892`
- **Type**: Integer

### RESULT_HEIGHT

- **Description**: Defines the height of the resulting image.
- **Default Value**: 1206
- **Usage**: `RESULT_HEIGHT=1206`
- **Type**: Integer

### PADDING

- **Description**: Defines the padding around the image.
- **Default Value**: 0
- **Usage**: `PADDING=0`
- **Type**: Integer

### BORDER

- **Description**: Defines the border width around the cropped image.
- **Default Value**: 30
- **Usage**: `BORDER=30`
- **Type**: Integer

### FRAME_SCALE_FACTOR

- **Description**: Defines the scale factor for resizing the frame image.
- **Default Value**: 0.3
- **Usage**: `FRAME_SCALE_FACTOR=0.3`
- **Type**: Float