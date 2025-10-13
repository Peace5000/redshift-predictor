from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import os
import re

app = Flask(__name__)
CORS(app)

# In-memory storage for uploaded data
data_store = {
    "metadata": None,
    "spectra": None,
    "knowledge_base": ""
}

# Load knowledge base from file on startup
def load_knowledge_base():
    """Load knowledge from a text file in the project folder"""
    knowledge_file = "astro.txt"  # Change this to your filename
    
    if os.path.exists(knowledge_file):
        with open(knowledge_file, 'r', encoding='utf-8') as f:
            data_store["knowledge_base"] = f.read()
        print(f"✅ Knowledge base loaded from {knowledge_file}")
        print(f"   Content length: {len(data_store['knowledge_base'])} characters")
    else:
        print(f"⚠️ Knowledge file '{knowledge_file}' not found. Create it in the project folder.")
        data_store["knowledge_base"] = ""

# Load knowledge base when app starts
load_knowledge_base()

# Serve the HTML file
@app.route("/")
def index():
    return send_from_directory('.', 'index.html')

@app.route("/upload", methods=["POST"])
def upload_files():
    files = request.files
    if "metadata" in files:
        data_store["metadata"] = np.load(files["metadata"])
        print(f"Metadata loaded: shape {data_store['metadata'].shape}")
    if "spectra" in files:
        data_store["spectra"] = np.load(files["spectra"])
        print(f"Spectra loaded: shape {data_store['spectra'].shape}")
    return jsonify({"status": "ok"})

def create_prediction_plot(y_test, y_pred, title, r2):
    """Helper function to create prediction vs actual plot"""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Redshift', fontsize=12)
    plt.ylabel('Predicted Redshift', fontsize=12)
    plt.title(f'{title}\n(R² = {r2:.4f})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{plot_base64}"

@app.route("/predict", methods=["POST"])
def predict_redshift():
    if data_store["metadata"] is None or data_store["spectra"] is None:
        return jsonify({"error": "Missing files"}), 400
    
    try:
        metadata = data_store["metadata"]
        spectra = data_store["spectra"]
        
        print(f"Running predictions on {spectra.shape[0]} samples...")
        
        # 1. PCA transform
        pca = PCA(n_components=2)
        transformed_spectra = pca.fit_transform(spectra)
        print(f"PCA completed: {transformed_spectra.shape}")
        
        # 2. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            transformed_spectra, metadata[:,0], test_size=0.33, random_state=42
        )
        print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
        
        # ==================== RANDOM FOREST MODEL ====================
        print("Training Random Forest...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        rf_mse = mean_squared_error(y_test, rf_pred)
        rf_r2 = r2_score(y_test, rf_pred)
        print(f"Random Forest - MSE: {rf_mse:.6f}, R²: {rf_r2:.4f}")
        
        # Create Random Forest plot
        rf_plot = create_prediction_plot(y_test, rf_pred, "Random Forest Regressor", rf_r2)
        
        # ==================== NEURAL NETWORK MODEL ====================
        print("Training Neural Network...")
        # Scale features for neural network
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        nn_model = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=1)
        nn_model.fit(X_train_scaled, y_train)
        nn_pred = nn_model.predict(X_test_scaled)
        
        nn_mse = mean_squared_error(y_test, nn_pred)
        nn_r2 = r2_score(y_test, nn_pred)
        print(f"Neural Network - MSE: {nn_mse:.6f}, R²: {nn_r2:.4f}")
        
        # Create Neural Network plot
        nn_plot = create_prediction_plot(y_test, nn_pred, "Neural Network (MLP)", nn_r2)
        
        # Return results to frontend
        response = {
            "totalSamples": int(spectra.shape[0]),
            "trainSamples": int(X_train.shape[0]),
            "testSamples": int(X_test.shape[0]),
            "avgRedshift": float(np.mean(metadata[:,0])),
            "stdRedshift": float(np.std(metadata[:,0])),
            "minRedshift": float(np.min(metadata[:,0])),
            "maxRedshift": float(np.max(metadata[:,0])),
            
            # Random Forest results
            "rf_accuracy": float(rf_r2 * 100),
            "rf_r2": float(rf_r2),
            "rf_mse": float(rf_mse),
            "rf_plot": rf_plot,
            
            # Neural Network results
            "nn_accuracy": float(nn_r2 * 100),
            "nn_r2": float(nn_r2),
            "nn_mse": float(nn_mse),
            "nn_plot": nn_plot,
            
            "pcaComponents": transformed_spectra.tolist()
        }
        
        print("Prediction completed successfully!")
        return jsonify(response)
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def is_greeting(message):
    """Check if message is a greeting using word boundaries"""
    greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    message_lower = message.lower().strip()
    
    # Check if message is exactly a greeting
    if message_lower in greetings:
        return True
    
    # Check if message starts with greeting followed by punctuation or space
    for greeting in greetings:
        # Use regex for word boundary matching
        pattern = r'\b' + re.escape(greeting) + r'\b'
        if re.match(pattern, message_lower):
            # Make sure it's at the start and followed by space, punctuation, or end of string
            if message_lower == greeting or message_lower.startswith(greeting + ' ') or \
               message_lower.startswith(greeting + ',') or message_lower.startswith(greeting + '!'):
                return True
    
    return False

def search_in_knowledge(query, context_lines=5):
    """
    Enhanced search for relevant information in the knowledge base
    Returns ranked results based on relevance
    """
    knowledge = data_store["knowledge_base"]
    
    if not knowledge:
        return None
    
    query_lower = query.lower()
    lines = knowledge.split('\n')
    
    # Extract keywords from query (ignore common words)
    stop_words = {'what', 'how', 'does', 'the', 'is', 'are', 'can', 'you', 'tell', 'me', 
                  'about', 'explain', 'with', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'a', 'an'}
    keywords = [word.strip('.,!?;:') for word in query_lower.split() 
                if len(word) > 2 and word not in stop_words]
    
    if not keywords:
        return None
    
    # Find relevant sections with scoring
    scored_sections = []
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        score = 0
        
        # Score based on keyword matches
        for keyword in keywords:
            if keyword in line_lower:
                # Exact word match gets higher score
                if re.search(r'\b' + re.escape(keyword) + r'\b', line_lower):
                    score += 10
                else:
                    score += 5
        
        # If line has a good score, include context around it
        if score > 0:
            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)
            context = '\n'.join(lines[start:end]).strip()
            
            # Avoid duplicates
            if context not in [s[1] for s in scored_sections]:
                scored_sections.append((score, context))
    
    if not scored_sections:
        return None
    
    # Sort by score (highest first) and return top results
    scored_sections.sort(reverse=True, key=lambda x: x[0])
    
    # Return top 2-3 most relevant sections
    top_sections = [section[1] for section in scored_sections[:2]]
    return '\n\n---\n\n'.join(top_sections)

def extract_main_topic(message):
    """Extract the main topic from the user's question"""
    message_lower = message.lower()
    
    # Common question patterns
    patterns = {
        'redshift': ['redshift', 'red shift', 'doppler'],
        'pca': ['pca', 'principal component', 'dimensionality reduction'],
        'random forest': ['random forest', 'forest', 'decision tree'],
        'neural network': ['neural network', 'nn', 'mlp', 'deep learning'],
        'training': ['training', 'train', 'model training', 'learning'],
        'spectra': ['spectra', 'spectral', 'spectrum', 'wavelength'],
        'accuracy': ['accuracy', 'r2', 'mse', 'performance', 'score'],
        'dataset': ['dataset', 'data', 'samples'],
    }
    
    for topic, keywords in patterns.items():
        for keyword in keywords:
            if keyword in message_lower:
                return topic
    
    return None

@app.route("/chat", methods=["POST"])
def chat():
    """Enhanced AI Chatbot that answers questions from the knowledge base"""
    data = request.json
    message = data.get("message", "").strip()
    
    if not message:
        return jsonify({"response": "Please ask me a question!"})
    
    print(f"📩 Chat received: {message}")
    
    # Handle greetings with improved detection
    if is_greeting(message):
        return jsonify({
            "response": """👋 Hello! I'm your astronomy AI assistant!

I can help you with:
🔭 Questions about redshift and astronomy
🤖 Machine learning models and training
📊 Understanding your spectral data
💡 Explaining the prediction process

What would you like to know about?"""
        })
    
    # Check if knowledge base is loaded
    if not data_store["knowledge_base"]:
        return jsonify({
            "response": "⚠️ No knowledge base found. Please create 'astro.txt' in the project folder with astronomy information!"
        })
    
    # Handle help commands
    message_lower = message.lower()
    if message_lower in ["help", "what can you do", "what can you do?"]:
        return jsonify({
            "response": """**I can help you with:**

🔭 Answer questions about astronomy and redshift
📚 Search through my knowledge base
💡 Explain the training process and models
📊 Provide information about your data

**Try asking:**
- "What is redshift?"
- "How does PCA work?"
- "Explain Random Forest"
- "What's the training process?"
- "What is spectral analysis?" """
        })
    
    # Extract main topic and search
    topic = extract_main_topic(message)
    response_text = ""
    
    # Try searching with the full query first
    print(f"🔍 Searching knowledge base for: {message}")
    result = search_in_knowledge(message)
    
    if result:
        response_text = f"**Here's what I found:**\n\n{result}"
        print(f"✅ Found relevant information")
    
    # If still no result, try searching for the detected topic
    if not response_text and topic:
        print(f"🔍 Searching for topic: {topic}")
        result = search_in_knowledge(topic, context_lines=8)
        if result:
            response_text = f"**Here's what I found about {topic}:**\n\n{result}"
            print(f"✅ Found information about {topic}")
    
    # Ultimate fallback
    if not response_text:
        print(f"❌ No information found")
        response_text = f"""I couldn't find specific information about "{message}" in my knowledge base.

**Try asking:**
- More specific questions
- Questions about redshift, PCA, Random Forest, or Neural Networks
- Type "help" to see what I can do

📚 My knowledge base has {len(data_store['knowledge_base'])} characters loaded from astro.txt"""
    
    print(f"💬 Responding with {len(response_text)} characters")
    return jsonify({"response": response_text})

if __name__ == "__main__":
    app.run(debug=True, port=5000)