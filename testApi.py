import joblib
from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

with open("Placement.joblib", 'rb') as f:
    model = joblib.load(f)

with open("Job.joblib", 'rb') as f:
    model2 = joblib.load(f)

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict/course', methods=['POST'])
def predictCourse():
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        f2 = pd.read_csv(file)
        prediction = model.predict(f2).tolist()
        return jsonify({'prediction': prediction})
    else:
        return jsonify({'error': 'Invalid file type. Only CSV files are allowed.'}), 400


@app.route('/predict/job', methods=['POST'])
def predictJob():
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        f2 = pd.read_csv(file)
        prediction = model2.predict(f2).tolist()
        return jsonify({'prediction': prediction})
    else:
        return jsonify({'error': 'Invalid file type. Only CSV files are allowed.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
