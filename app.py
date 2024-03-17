from flask import Flask, render_template, request, redirect, flash
from pytorch_lightning import Trainer
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import os
import torch
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load OCR model and processor
model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-large-handwritten"
).to("cuda" if torch.cuda.is_available() else "cpu")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")


# Function to check if file extension is allowed
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# OCR function
def ocr(image):
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


# Homepage
@app.route("/")
def home():
    return render_template("index.html")


# OCR endpoint
@app.route("/upload", methods=["POST"])
def upload_file():
    if request.method == "POST":
        # Check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # Save the uploaded image
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            # Perform OCR
            image = Image.open(file_path).convert("RGB")
            text = ocr(image)
            return render_template("index.html", text=text)
        else:
            flash("Invalid file format. Allowed formats are: png, jpg, jpeg, gif")
            return redirect(request.url)


if __name__ == "__main__":
    app.run(debug=True)
