# app.py

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from OpenPoseImage import process_image, base_path
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploaded_images'
PROCESSED_FOLDER = 'processed_images'

app.config['UPLOAD_FOLDER'] = os.path.join(base_path, UPLOAD_FOLDER)
app.config['STATIC_FOLDER'] = os.path.join(base_path, 'static')
app.config['PROCESSED_FOLDER'] = os.path.join(app.config['STATIC_FOLDER'], PROCESSED_FOLDER)

# Asegúrate de que las carpetas existan
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                processed_image_name, posture = process_image(filepath)
                processed_image_url = url_for('static', filename=os.path.join(PROCESSED_FOLDER, processed_image_name).replace("\\", "/"))
            except Exception as e:
                print(f'Error processing image: {e}')
                raise e
            
            return redirect(url_for('results', image_path=processed_image_url, posture=posture))
    return render_template('index.html')

@app.route('/results/<path:image_path>/<posture>')
def results(image_path, posture):
    return render_template('results.html', image_path=image_path, posture=posture)

if __name__ == '__main__':
    app.run(debug=True)
