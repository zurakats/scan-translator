from flask import Flask, request, send_file, render_template
from flask_cors import CORS
from image_processor import process_image
from werkzeug.utils import secure_filename
import os

# Inisialisasi Flask dan aktifkan CORS
app = Flask(__name__)
CORS(app)

# Direktori upload
UPLOAD_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Route untuk halaman utama
@app.route('/')
def home():
    return render_template('index.html')  # pastikan ada di folder templates/

# Route untuk proses gambar
@app.route('/process-image', methods=['POST'])
def process_image_endpoint():
    try:
        if 'image' not in request.files:
            return {'error': 'No image provided'}, 400

        file = request.files['image']
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Proses gambar dengan model
        output_img = process_image(filepath)

        # Simpan hasil ke file baru
        output_path = os.path.join(UPLOAD_FOLDER, 'output.png')
        output_img.save(output_path)

        return send_file(output_path, mimetype='image/png')

    except Exception as e:
        print(f"⚠️ Error saat memproses gambar: {e}")
        return {'error': str(e)}, 500

# Jalankan server lokal
if __name__ == '__main__':
    app.run(debug=True)
