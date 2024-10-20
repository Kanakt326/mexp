Expiry & Manufacturing Date Validation (YOLOv5 + PaddleOCR)

This project focuses on validating Expiry and Manufacturing dates from product packaging. The system employs a YOLOv5 model to detect regions on packaging that may contain date information, followed by detailed text extraction using PaddleOCR. Post-extraction, the dates are validated using regular expressions (regex) to ensure they conform to standard formats.
Requirements.

To run the project, you will need the following dependencies:
* Python 3.7+
* PyTorch
* YOLOv5
* PaddleOCR
* OpenCV
* Numpy
* Regex (re)

  Install the required Python packages via requirements.txt:
  pip install -r requirements.txt
  Run the Streamlit App
  streamlit run app.py


Key Libraries:
YOLOv5: For object detection, identifying potential regions containing date information on packaging.
PaddleOCR: For Optical Character Recognition (OCR) to extract text from detected regions.
Regex: For date format validation.

Model Overview
YOLOv5 (PyTorch): Trained or fine-tuned to detect regions that potentially contain date information on product packaging, such as Expiry Date and Manufacturing Date.
PaddleOCR: Once the regions are detected and cropped, PaddleOCR extracts the text for further processing.

Process Flow
1. Detection of Date Regions
The YOLOv5 model processes an input image and detects regions where date information is likely located.
2. Cropping Detected Regions
Post-detection, the regions are cropped from the original image to isolate areas that may contain date information.
3. Text Extraction
PaddleOCR processes the cropped regions to extract text, focusing on dates.
4. Date Validation
Extracted text is passed through a regex-based validation system, which identifies whether the text conforms to standard date formats.
Regex Date Validation
The regex system supports multiple common date formats. Some of the formats that are validated include:
DD/MM/YYYY (e.g., 25/12/2023)
MM/DD/YYYY (e.g., 12/25/2023)
YYYY-MM-DD (e.g., 2023-12-25)
MMM DD, YYYY (e.g., Dec 25, 2023)
The regex expressions for these formats are designed to ensure only valid dates are captured.
![image](https://github.com/user-attachments/assets/34a30bc1-5664-4d9e-ba52-1e07b516039d)
   



