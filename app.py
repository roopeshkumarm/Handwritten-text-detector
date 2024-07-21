from flask import Flask, render_template, request, redirect, flash
from pytorch_lightning import Trainer
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import os
import torch
from werkzeug.utils import secure_filename

app = Flask(__name__)
# app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-large-handwritten"
).to("cuda" if torch.cuda.is_available() else "cpu")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")



def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS



def ocr(image):
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text



@app.route("/")
def home():
    return render_template("index.html")



@app.route("/upload", methods=["POST"])
def upload_file():
    if request.method == "POST":
        
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]

        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            image = Image.open(file_path).convert("RGB")
            text = ocr(image)
            return render_template("index.html", text=text)
        else:
            flash("Invalid file format. Allowed formats are: png, jpg, jpeg, gif")
            return redirect(request.url)


if __name__ == "__main__":
    app.run(debug=True)
