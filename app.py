from flask import Flask, request, jsonify
import pickle
import re
import os
import sys
import logging
import importlib
import traceback

# Configurar logging detallado
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/deka/mysite/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Verificar el entorno
logger.info("=== Verificación del Entorno ===")
logger.info(f"Python Version: {sys.version}")
logger.info(f"Working Directory: {os.getcwd()}")

# Importar dependencias con verificación
dependencies = ['numpy', 'scipy', 'sklearn']
for dep in dependencies:
    try:
        module = importlib.import_module(dep)
        logger.info(f"{dep} version: {module.__version__}")
    except Exception as e:
        logger.error(f"Error importing {dep}: {str(e)}")

# Definir rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'lr_best.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')

logger.info(f"Base Directory: {BASE_DIR}")
logger.info(f"Model Path: {MODEL_PATH}")
logger.info(f"Vectorizer Path: {VECTORIZER_PATH}")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else "no_text"

# Cargar modelos con manejo detallado de errores
def load_models():
    global model, vectorizer, models_loaded
    try:
        logger.info("=== Iniciando carga de modelos ===")

        # Verificar existencia de archivos
        for path, name in [(MODEL_PATH, 'Model'), (VECTORIZER_PATH, 'Vectorizer')]:
            if os.path.exists(path):
                size = os.path.getsize(path)
                logger.info(f"{name} file exists. Size: {size} bytes")
            else:
                logger.error(f"{name} file not found at {path}")
                return False

        # Intentar cargar vectorizador
        logger.info("Loading vectorizer...")
        with open(VECTORIZER_PATH, 'rb') as f:
            try:
                vectorizer = pickle.load(f)
                logger.info(f"Vectorizer type: {type(vectorizer)}")
            except Exception as e:
                logger.error(f"Error unpickling vectorizer: {str(e)}")
                logger.error(traceback.format_exc())
                return False

        # Intentar cargar modelo
        logger.info("Loading model...")
        with open(MODEL_PATH, 'rb') as f:
            try:
                model = pickle.load(f)
                logger.info(f"Model type: {type(model)}")
            except Exception as e:
                logger.error(f"Error unpickling model: {str(e)}")
                logger.error(traceback.format_exc())
                return False

        # Verificar objetos cargados
        if model is None or vectorizer is None:
            logger.error("One or both models loaded as None")
            return False

        logger.info("=== Modelos cargados exitosamente ===")
        return True

    except Exception as e:
        logger.error(f"Unexpected error loading models: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Variables globales
model = None
vectorizer = None
models_loaded = load_models()

@app.route('/reload', methods=['POST'])
def reload_models():
    """Endpoint para recargar los modelos manualmente"""
    global models_loaded
    models_loaded = load_models()
    return jsonify({
        'success': models_loaded,
        'message': 'Models reloaded successfully' if models_loaded else 'Failed to reload models'
    })

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    if not models_loaded:
        return jsonify({
            'error': 'Models not loaded',
            'details': 'Service initialization incomplete'
        }), 503

    try:
        data = request.get_json()
        if not data or 'review' not in data:
            return jsonify({
                'error': 'No review provided',
                'usage': {'example': {'review': 'Your review text here'}}
            }), 400

        processed_text = clean_text(data['review'])
        text_vectorized = vectorizer.transform([processed_text])
        prediction = model.predict(text_vectorized)
        prediction_proba = model.predict_proba(text_vectorized)

        response = {
            'sentiment': prediction[0],
            'confidence': float(max(prediction_proba[0])),
            'processed_text': processed_text
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    environment_info = {
        'python_version': sys.version,
        'working_directory': os.getcwd(),
        'base_directory': BASE_DIR,
        'files_exist': {
            'model': os.path.exists(MODEL_PATH),
            'vectorizer': os.path.exists(VECTORIZER_PATH)
        },
        'file_sizes': {
            'model': os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0,
            'vectorizer': os.path.getsize(VECTORIZER_PATH) if os.path.exists(VECTORIZER_PATH) else 0
        }
    }

    for dep in dependencies:
        try:
            module = importlib.import_module(dep)
            environment_info[f'{dep}_version'] = module.__version__
        except:
            environment_info[f'{dep}_version'] = 'not found'

    return jsonify({
        'status': 'healthy' if models_loaded else 'degraded',
        'models_loaded': models_loaded,
        'environment': environment_info
    })

if __name__ == '__main__':
    app.run(debug=True)
