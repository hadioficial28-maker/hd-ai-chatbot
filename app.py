from flask import Flask, render_template, request, jsonify, send_from_directory
from openai import OpenAI
import os
import urllib.parse
import time
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

app = Flask(__name__, static_folder='static', static_url_path='/static')

# ==========================================
# CONFIGURATION
# ==========================================

# Upload folder setup
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Get API keys from environment variables (SECURE!)
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
FIREBASE_CONFIG = {
    'apiKey': os.getenv('FIREBASE_API_KEY'),
    'authDomain': os.getenv('FIREBASE_AUTH_DOMAIN'),
    'projectId': os.getenv('FIREBASE_PROJECT_ID'),
    'storageBucket': os.getenv('FIREBASE_STORAGE_BUCKET'),
    'messagingSenderId': os.getenv('FIREBASE_MESSAGING_SENDER_ID'),
    'appId': os.getenv('FIREBASE_APP_ID')
}

# Check if API key exists
if not OPENROUTER_API_KEY:
    print("⚠️ WARNING: OPENROUTER_API_KEY not found in environment variables!")
    print("Please set it in .env file or Render environment variables")

# Initialize OpenAI client
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# ==========================================
# AVAILABLE AI MODELS
# ==========================================
AVAILABLE_MODELS = {
    "llama31-70b": {
        "name": "Llama 3.1 (70B)",
        "provider": "openrouter",
        "id": "meta-llama/llama-3.1-70b-instruct"
    },
    "llama31-8b": {
        "name": "Llama 3.1 (8B) - Fast",
        "provider": "openrouter",
        "id": "meta-llama/llama-3.1-8b-instruct"
    },
    "qwen25-coder": {
        "name": "Qwen 2.5 Coder (32B)",
        "provider": "openrouter",
        "id": "qwen/qwen-2.5-coder-32b-instruct"
    },
    "qwen25-72b": {
        "name": "Qwen 2.5 (72B)",
        "provider": "openrouter",
        "id": "qwen/qwen-2.5-72b-instruct"
    },
    "mistral-nemo": {
        "name": "Mistral Nemo",
        "provider": "openrouter",
        "id": "mistralai/mistral-nemo"
    },
    "phi4": {
        "name": "Phi-4 (14B)",
        "provider": "openrouter",
        "id": "microsoft/phi-4"
    }
}

DEFAULT_MODEL = "llama31-8b"

# ==========================================
# FILE UPLOAD CONFIGURATION
# ==========================================
ALLOWED_EXTENSIONS = {
    'image': {'png', 'jpg', 'jpeg', 'gif', 'webp'},
    'video': {'mp4', 'avi', 'mov', 'webm', 'mkv'},
    'audio': {'mp3', 'wav', 'ogg', 'm4a'},
    'document': {'pdf', 'txt', 'doc', 'docx'}
}

def allowed_file(filename, file_type=None):
    """Check if file extension is allowed"""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if file_type:
        return ext in ALLOWED_EXTENSIONS.get(file_type, set())
    all_allowed = set()
    for exts in ALLOWED_EXTENSIONS.values():
        all_allowed.update(exts)
    return ext in all_allowed

def get_file_type(filename):
    """Get file type from filename"""
    if '.' not in filename:
        return 'unknown'
    ext = filename.rsplit('.', 1)[1].lower()
    for ftype, extensions in ALLOWED_EXTENSIONS.items():
        if ext in extensions:
            return ftype
    return 'unknown'

# ==========================================
# ROUTES
# ==========================================

@app.route('/')
def home():
    """Home page - renders index.html"""
    return render_template('index.html', models=AVAILABLE_MODELS, default_model=DEFAULT_MODEL)

@app.route('/chat', methods=['POST'])
def chat_with_bot():
    """Handle chat requests with AI models"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        selected_model_key = data.get('model', DEFAULT_MODEL)
        
        model_info = AVAILABLE_MODELS.get(selected_model_key, AVAILABLE_MODELS[DEFAULT_MODEL])
        model_id = model_info['id']
        
        if not user_message:
            return jsonify({'response': 'Kuch likh kar toh bhejo!'})

        # Strong system prompt for better code generation
        system_prompt = """You are an expert AI programmer and software engineer. 
        Your task is to write high-quality, complete, and working code.
        
        RULES:
        1. ALWAYS provide COMPLETE code. Never use placeholders.
        2. Use proper markdown code blocks (e.g., ```cpp ... ```).
        3. Add clear comments explaining the logic.
        4. If the user asks for a game or complex app, write the full logic.
        5. Respond in the same language the user used (English, Urdu, or Hindi).
        6. If there is an error, politely explain it and suggest a fix.
        7. For non-code questions, be helpful and friendly.
        """

        response = openrouter_client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=4096,  # Increased for long code
            temperature=0.7
        )
        
        bot_reply = response.choices[0].message.content
        return jsonify({'response': bot_reply})
    
    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({'response': f'Error: {str(e)}'}), 200

@app.route('/generate-image', methods=['POST'])
def generate_image():
    """Generate images using Pollinations.ai (FREE)"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        encoded_prompt = urllib.parse.quote(prompt)
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true&seed={int.from_bytes(os.urandom(4), byteorder='big')}"
        
        return jsonify({
            'success': True,
            'image_url': image_url,
            'prompt': prompt
        })
    
    except Exception as e:
        print(f"Image generation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        file_type = request.form.get('type', 'auto')
        if file_type == 'auto':
            file_type = get_file_type(file.filename)
        
        if not allowed_file(file.filename, file_type):
            return jsonify({
                'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS.get(file_type, set()))}'
            }), 400
        
        # Check file size (max 50MB)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > 50 * 1024 * 1024:
            return jsonify({'error': 'File too large. Max size: 50MB'}), 400
        
        # Save file with timestamp
        timestamp = int(time.time())
        original_filename = file.filename
        safe_filename = f"{timestamp}_{original_filename.replace(' ', '_')}"
        filepath = os.path.join(UPLOAD_FOLDER, safe_filename)
        file.save(filepath)
        
        file_url = f"/static/uploads/{safe_filename}"
        
        return jsonify({
            'success': True,
            'filename': original_filename,
            'saved_filename': safe_filename,
            'url': file_url,
            'type': file_type,
            'size_mb': round(file_size / (1024 * 1024), 2)
        })
    
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(UPLOAD_FOLDER, filename)

# ==========================================
# MAIN
# ==========================================
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 HD AI Chatbot Server Starting...")
    print("="*50)
    print("📍 URL: http://127.0.0.1:5000")
    print("\n✨ Features:")
    print("  💬 AI Chat (6 Models)")
    print("  🖼️ Image Generation (FREE)")
    print("  📎 File Upload (Images, Videos, Audio, Documents)")
    print("\n✅ Sab kuch FREE hai!")
    print("="*50 + "\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')