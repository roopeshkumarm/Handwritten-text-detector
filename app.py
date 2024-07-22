from flask import Flask, request, jsonify
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
from flask_cors import CORS
import io

app = Flask(__name__)
CORS(app)

# Load OCR model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten").to(device)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")

# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# OCR function
def ocr(image):
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

# Test endpoint
@app.route("/", methods=["GET"])
def get_method():
    return jsonify({"msg": "working..."})

# OCR endpoint
@app.route("/api/ocr", methods=["POST"])
def upload_file():
    app.logger.info(f"Request Data: {request.data}")
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        # Perform OCR
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        text = ocr(image)
        return jsonify({"text": text})
    else:
        return jsonify({"error": "Invalid file format. Allowed formats are: png, jpg, jpeg, gif"}), 400

if __name__ == "__main__":
    app.run(debug=True)
