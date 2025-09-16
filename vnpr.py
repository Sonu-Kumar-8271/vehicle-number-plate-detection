import cv2
import pytesseract
import pandas as pd
import datetime
import os
import re
import glob
import threading
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox

# =============== CONFIGURATION ===============
TESSERACT_PATH = r"/opt/homebrew/bin/tesseract"  # Change this if needed
YOLO_MODEL_PATH = "Model/license_plate_detector.pt"
YOLO_CONF_THRESHOLD = 0.3
MIN_PLATE_LENGTH = 5

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# =============== INIT ===============
today = datetime.datetime.now().strftime("%Y-%m-%d")
plate_dir = f"detected_plates_{today}"
full_frame_dir = f"detected_frames_{today}"

os.makedirs(plate_dir, exist_ok=True)
os.makedirs(full_frame_dir, exist_ok=True)

try:
    model = YOLO(YOLO_MODEL_PATH)
except Exception as e:
    tk.Tk().withdraw()
    messagebox.showerror("Model Load Error", f"Failed to load YOLO model:\n{str(e)}")
    exit(1)

detected_plates = []
lock = threading.Lock()

# =============== FUNCTION TO PROCESS FRAME ===============
def process_frame(frame, source_name="unknown"):
    global detected_plates
    try:
        results = model.predict(frame, conf=YOLO_CONF_THRESHOLD)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                plate_img = frame[y1:y2, x1:x2]

                try:
                    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    dilated = cv2.dilate(thresh, np.ones((2, 2), np.uint8), iterations=1)

                    config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    text = pytesseract.image_to_string(dilated, config=config).strip()
                    clean_text = re.sub(r'[^A-Za-z0-9-]', '', text)

                    if len(clean_text) < MIN_PLATE_LENGTH:
                        continue

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, clean_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                    with lock:
                        if clean_text not in [p[0] for p in detected_plates]:
                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            print(f"Detected: {clean_text} from {source_name} at {timestamp}")
                            detected_plates.append((clean_text, timestamp, source_name))

                            plate_filename = f"{plate_dir}/plate_{clean_text}_{timestamp.replace(' ', '_').replace(':', '')}.jpg"
                            cv2.imwrite(plate_filename, plate_img)

                except Exception as e:
                    print(f"OCR Error: {e}")
    except Exception as e:
        print(f"Model prediction failed: {e}")

# =============== SCAN FROM WEBCAM ===============
def scan_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        process_frame(frame, source_name="webcam")

        filename = f"{full_frame_dir}/frame_webcam_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)

        try:
            cv2.imshow("Webcam - Press 'q' to Quit", frame)
        except cv2.error as e:
            print("cv2.imshow() error:", e)
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    save_results()

# =============== SCAN FROM IMAGES ===============
def scan_images():
    folder_selected = filedialog.askdirectory()
    if not folder_selected:
        return

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]:
        image_paths.extend(glob.glob(os.path.join(folder_selected, ext)))

    for path in image_paths:
        frame = cv2.imread(path)
        if frame is None:
            print(f"Can't read image: {path}")
            continue

        process_frame(frame, source_name=os.path.basename(path))

        out_path = os.path.join(full_frame_dir, f"processed_{os.path.basename(path)}")
        cv2.imwrite(out_path, frame)

    messagebox.showinfo("Done", f"Processed {len(image_paths)} images.")
    save_results()

# =============== SAVE DETECTIONS ===============
def save_results():
    with lock:
        if detected_plates:
            df = pd.DataFrame(detected_plates, columns=["Plate Number", "Time", "Source"])
            out_path = f"detected_plates_{today}.xlsx"
            df.to_excel(out_path, index=False)
            print(f"Saved to {out_path}")
            messagebox.showinfo("Saved", f"Results saved to {out_path}")
        else:
            print("No plates to save.")
            messagebox.showwarning("No Data", "No plates were detected to save.")

# =============== CLEAR TODAY'S FILES ===============
def clear_today_files():
    try:
        for folder in [plate_dir, full_frame_dir]:
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    os.remove(os.path.join(folder, f))
                os.rmdir(folder)

        excel_path = f"detected_plates_{today}.xlsx"
        if os.path.exists(excel_path):
            os.remove(excel_path)

        messagebox.showinfo("Cleared", "Today's files have been deleted.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to clear files: {e}")

# =============== TKINTER GUI ===============
root = tk.Tk()
root.title("Vehicle Number Plate Detector")
root.geometry("500x500")
root.configure(bg="#101b29")

style = {"font": ("Arial", 14), "width": 25, "padx": 10, "pady": 10}

tk.Label(root, text="Vehicle Number Plate Detector", font=("Arial", 20, "bold"),
         fg="#130404", bg="#00eeff").pack(pady=20)

tk.Button(root, text="📷 Scan from Webcam", command=scan_webcam,
          bg="#92f7ff", fg="#0d092a", **style).pack()

tk.Button(root, text="🖼️ Scan Image Folder", command=lambda: threading.Thread(target=scan_images).start(),
          bg="#35cfdb", fg="#0d092a", **style).pack()

tk.Button(root, text="📅 Save Results to Excel", command=save_results,
          bg="#a6abff", fg="#0d092a", **style).pack()

tk.Button(root, text="🗑️ Clear Today's Files", command=clear_today_files,
          bg="#3f3fd8", fg="#000000", **style).pack()

tk.Button(root, text="❌ Exit", command=root.quit,
          bg="#0d092a", fg="#000000", **style).pack(pady=10)

root.mainloop()
