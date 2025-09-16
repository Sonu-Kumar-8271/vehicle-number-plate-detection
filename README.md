# 🚘 Vehicle Number Plate Detection (YOLOv8 + OCR)

An Automatic Number Plate Recognition (ANPR) system built with YOLOv8 for vehicle plate detection and OCR (Tesseract) for text extraction.
The project includes a Tkinter-based GUI for real-time interaction, webcam scanning, image folder processing, and saving detected results to Excel.

✨ Features

YOLOv8 Detection → Detects license plates from images, videos, and webcam streams

OCR Integration → Uses Tesseract to extract alphanumeric text from plates

GUI with Tkinter →

Scan from webcam

Process a folder of images

Save detections to Excel

Clear today’s results

Preprocessing → Grayscale, resizing, thresholding, and dilation for improved OCR

Auto-Save → Cropped plates and annotated frames saved with timestamps

🛠️ Tech Stack

Python 3.x

YOLOv8 (Ultralytics)

OpenCV

Tesseract OCR

Tkinter (GUI)

Pandas & Numpy

📦 Installation

Clone this repository

git clone https://github.com/your-username/vehicle-number-plate-detection.git
cd vehicle-number-plate-detection


Create a virtual environment (recommended)

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows


Install dependencies

pip install -r requirements.txt


Download YOLOv8 model

Place your trained YOLOv8 model at:

Model/license_plate_detector.pt


Set Tesseract path

Update the path in vnpr.py:

TESSERACT_PATH = r"/opt/homebrew/bin/tesseract"

▶️ Usage

Run the application:

python vnpr.py

GUI Options:

📷 Scan from Webcam → Detect plates live from camera

🖼️ Scan Image Folder → Select a folder of images for batch detection

📅 Save Results to Excel → Save detected plate numbers with timestamp

🗑️ Clear Today’s Files → Delete saved plates and frames for today

📂 Output

Cropped plate images → detected_plates_YYYY-MM-DD/

Annotated frames → detected_frames_YYYY-MM-DD/

Excel results → detected_plates_YYYY-MM-DD.xlsx

🎯 Use Cases

Automated Toll Collection

Parking Management Systems

Traffic Monitoring & Law Enforcement

Smart City Applications

🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you’d like to change.

📜 License

This project is licensed under the MIT License.
