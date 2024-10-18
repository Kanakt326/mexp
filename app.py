import streamlit as st
import cv2
import re
import numpy as np  # Import NumPy
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Load the trained YOLOv8 model
model = YOLO('best (2).pt')

# List of class names
class_names = ['Label', 'Product', 'Quantity', 'brand', 'detectionofpackage', 'flavor', 'other_info']

# Function to extract multiple dates
def extract_dates(text):
    # Patterns for different date formats
    date_patterns = [
        r'\b(0[1-9]|1[0-2])[-/](\d{2,4})\b',             # MM/YYYY or MM-YYYY
        r'\b(0[1-9]|1[0-2])(\d{2})\b',                    # MMYY (e.g., 1223)
        r'\b(\d{2})[-/](0[1-9]|1[0-2])[-/](\d{2})\b',     # dd/mm/yy
        r'\b(\d{2})[-/](0[1-9]|1[0-2])[-/](\d{4})\b',     # dd/mm/yyyy
        r'\b(0[1-9]|1[0-2])[-/](\d{2})\b',                # MM/yy
        r'\b(0[1-9]|1[0-2])[-/](\d{4})\b',                # MM/yyyy
        r'\b([A-Za-z]{3})[-/](\d{2})\b',                  # Mon/yy (e.g., AUG/24)
        r'\b([A-Za-z]{3})[-/](\d{4})\b',                  # Mon/yyyy (e.g., AUG/2024)
        r'\b([A-Za-z]{3,9})[-/](\d{2})\b',                # Month/yy (e.g., August/24)
        r'\b([A-Za-z]{3,9})[-/](\d{4})\b'                 # Month/yyyy (e.g., August/2024)
    ]
    
    month_abbreviations = {
        'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
        'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
        'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
    }

    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match) == 3:  # dd/mm/yyyy or dd/mm/yy
                day, month, year = match
                dates.append(f"{day}/{month}/{year}")  # Keep original format
            elif len(match) == 2:  # MM/YYYY or MM-YYYY or MMYY
                month, year = match
                dates.append(f"{month}/{year}")  # Keep original format
            elif len(match) == 1:  # Month in words
                month_word, year = match
                month_num = month_abbreviations.get(month_word.lower(), None)
                if month_num:
                    dates.append(f"{month_num}/{year}")  # Convert month name to number

    return dates  # Return list of formatted dates

# Updated function to extract the largest price from text
# Updated function to extract the largest price from text
def extract_prices(text):
    # Improved pattern to capture price variations, including MRP, Rs, and just numbers
    pattern = r'(?i)(?:MRP|(?:Rs\.?|₹))?\s*([\d,]+(?:\.\d{1,2})?)'
    
    # Find all matches in the text
    matches = re.findall(pattern, text)
    print("Matches found for prices:", matches)  # Debugging statement

    prices = []
    for price in matches:
        try:
            price_value = float(price.replace(',', ''))  # Convert to float
            prices.append(price_value)
        except ValueError:
            print(f"Warning: Could not convert '{price}' to float.")  # Debugging for conversion issues
    
    if prices:  # If we have valid prices, return the maximum
        return max(prices)
    return 'N/A'  # Return 'N/A' if no prices found

# Streamlit app
st.title("Product Information Extractor")

# List of predefined images
image_options = {
    "Image 1": "5.jpg",
    "Image 2": "2.jpg",
    "Image 3": "3.jpg",
    "Image 4": "4.jpg",
    "Image 5": "1.png",
    "Image 6": "6.jpg",
}

# Dropdown for selecting images
selected_image = st.selectbox("Select an image for testing:", list(image_options.keys()))

# File uploader
uploaded_file = st.file_uploader("Or upload an image from your local system...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
else:
    if selected_image:
        # Load the selected image
        image_path = image_options[selected_image]
        image = cv2.imread(image_path)

    if image is None:
        st.error("Could not load image.")
        st.stop()

# Run inference on the image
results = model(image, conf=0.25)  # Adjust the confidence threshold

# Variables to store extracted information
brand = 'N/A'
manufacturing_date = 'N/A'
expiry_date = 'N/A'
price = 'N/A'
extracted_texts = {}
cropped_images = []  # List to store cropped images

# Process the detected objects in the image
for result in results:
    boxes = result.boxes  # Get bounding boxes
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        class_name = class_names[class_id]

        # Crop the image using the bounding box
        cropped_img = image[y1:y2, x1:x2]
        cropped_images.append(cropped_img)  # Add cropped image to the list

        # Perform OCR on the cropped image
        ocr_result = ocr.ocr(cropped_img, cls=True)

        if ocr_result:
            extracted_text = " ".join([text_info[0] for line in ocr_result for _, text_info in line])
            extracted_texts[class_name] = extracted_text.strip()

            # Extract details if the class is 'brand' or 'Label'
            if class_name == 'brand':
                brand = extracted_text.strip()
            elif class_name == 'Label':
                extracted_dates = extract_dates(extracted_text)
                if len(extracted_dates) >= 2:
                    manufacturing_date = extracted_dates[0]
                    expiry_date = extracted_dates[1]
                elif len(extracted_dates) == 1:
                    manufacturing_date = extracted_dates[0]

                extracted_prices = extract_prices(extracted_text)
                if extracted_prices != 'N/A':
                    price = extracted_prices

# Display extracted information
st.subheader("Extracted Information")
st.write(f"Manufacturing Date: {manufacturing_date}")
st.write(f"Expiry Date: {expiry_date}")

# Display cropped images
st.subheader("Cropped Images")
for i, cropped_img in enumerate(cropped_images):
    st.image(cropped_img, caption=f"Cropped Image {i+1}", use_column_width=True)

# Display complete extracted texts
st.subheader("Extracted Texts from Detected Regions")
for class_name, text in extracted_texts.items():
    st.write(f"{class_name}: {text}")

# Display the selected image
st.image(image, caption="Selected Image", use_column_width=True)
