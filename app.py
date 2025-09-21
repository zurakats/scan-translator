from flask import Flask, request, send_file, render_template, jsonify
from flask_cors import CORS
from image_processor_v4 import process_image, selectOCR
from werkzeug.utils import secure_filename
import os
import base64

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process-image', methods=['POST'])
def process_image_endpoint():
    try:
        if 'image[]' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        sourceLang = request.form['source']
        targetLang = request.form['target']
        files = request.files.getlist('image[]')

        results = []
        ocr, index, translate_code = selectOCR(sourceLang)
        for file in files:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            output_img = process_image(filepath, ocr, index, translate_code, targetLang)

            output_path = os.path.join(UPLOAD_FOLDER, f"output_{filename}")
            output_img.save(output_path)

            with open(output_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')
            results.append(f"data:image/png;base64,{encoded}")

        return jsonify({'results': results})

    except Exception as e:
        print(f"⚠️ Error saat memproses gambar: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
