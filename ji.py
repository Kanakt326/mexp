from ultralytics import YOLO
import cv2
import os
from paddleocr import PaddleOCR

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # You can change the language as needed

# Load the trained YOLOv8 model
model = YOLO('C:\\Users\\kanak\\OneDrive\\Desktop\\Final\\best (2).pt')

# Load the image
image_path = 'C:\\Users\\kanak\\OneDrive\\Desktop\\Final\\images\\9.jpg'
image = cv2.imread(image_path)

# Run inference on the image
results = model(image_path, conf=0.25)  # Adjust the confidence threshold

# Output directory to save the cropped images
output_dir = 'C:\\Users\\kanak\\OneDrive\\Desktop\\Final\\inference_results\\crops\\'
os.makedirs(output_dir, exist_ok=True)

# List of class names
class_names = ['Label', 'Product', 'Quantity', 'brand', 'detectionofpackage', 'flavor', 'other_info']

# Dictionary to store extracted text results
extracted_texts = {}

# Loop through each detection result in the image
for result in results:
    boxes = result.boxes  # Get all bounding boxes
    for i, box in enumerate(boxes):
        # Get bounding box coordinates (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert coordinates to integers
        
        # Get class id and class name
        class_id = int(box.cls[0])
        class_name = class_names[class_id]

        # Crop the image using the bounding box
        cropped_img = image[y1:y2, x1:x2]

        # Save the cropped image
        crop_save_path = os.path.join(output_dir, f'{class_name}crop{i}.jpg')
        cv2.imwrite(crop_save_path, cropped_img)

        print(f'Cropped image saved at: {crop_save_path}')

        # Perform text extraction using PaddleOCR on the cropped image
        ocr_result = ocr.ocr(crop_save_path, cls=True)  # OCR with angle classification

        if ocr_result:  # Check if OCR returned any results
            extracted_text = ""
            for line in ocr_result:
                for box, text_info in line:
                    extracted_text += text_info[0] + " "

            # Store the extracted text for each cropped image
            extracted_texts[f'{class_name}crop{i}'] = extracted_text.strip()
            print(f'Extracted text from {class_name} crop {i}: {extracted_text.strip()}')
        else:
            print(f'No text detected in {class_name} crop {i}')

# Save the extracted text results to a file (optional)
with open(os.path.join(output_dir, 'extracted_texts.txt'), 'w') as f:
    for key, text in extracted_texts.items():
        f.write(f'{key}: {text}\n')

print('Text extraction completed.')

# Display predictions on the image (with bounding boxes)
for result in results:
    result.show()

# Optional: Save the result image with bounding boxes
for idx, result in enumerate(results):
    # Construct a filename for saving
    save_path = f'C:\\Users\\kanak\\OneDrive\\Desktop\\Final\\inference_result_{idx}.jpg'
    result.save(save_path) 