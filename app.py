import os
import json
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, session, jsonify
from werkzeug.utils import secure_filename
from QA import qa_pipeline
from Summarize import summarize_pipeline
from Keyword import keyword_pipeline

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Change this in production
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf', 'docx'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/api/settings', methods=['POST'])
def save_settings():
    data = request.json
    if 'api_key' in data:
        session['api_key'] = data['api_key']
        return jsonify({'status': 'success'})
    return jsonify({'error': 'No API key provided'}), 400

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        session['current_file'] = filepath
        return jsonify({'status': 'success', 'filename': filename})
    
    return jsonify({'error': 'Invalid file type'}), 400

def get_api_key():
    return session.get('api_key')

def get_current_file():
    return session.get('current_file')

@app.route('/api/questions', methods=['POST'])
def api_questions():
    api_key = get_api_key()
    if not api_key:
        return jsonify({'error': 'Please set your API Key in settings first.'}), 401
    
    filepath = get_current_file()
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'Please upload a file first.'}), 400

    try:
        results = qa_pipeline(filepath, api_key)
        return jsonify({'result': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/summarize', methods=['POST'])
def api_summarize():
    api_key = get_api_key()
    if not api_key:
        return jsonify({'error': 'Please set your API Key in settings first.'}), 401
    
    filepath = get_current_file()
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'Please upload a file first.'}), 400

    try:
        summary = summarize_pipeline(filepath, api_key)
        return jsonify({'result': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/keyword', methods=['POST'])
def api_keyword():
    api_key = get_api_key()
    if not api_key:
        return jsonify({'error': 'Please set your API Key in settings first.'}), 401
    
    filepath = get_current_file()
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'Please upload a file first.'}), 400

    try:
        keywords = keyword_pipeline(filepath, api_key)
        return jsonify({'result': keywords})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
