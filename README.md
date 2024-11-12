# API de Análisis de Sentimientos Proyecto M7 - Bootcamp Ciencia de Datos

Presentación: https://docs.google.com/presentation/d/1tRVpLlZKmuNpqcglKnY0-C4VBsXPdWpRXYcqJYJlF10/edit?usp=sharing

## Descripción General
Esta API proporciona análisis de sentimientos para reviews de aplicaciones, clasificándolas en positivas, neutrales o negativas.

## Endpoints

### Health Check
```
GET /health
```
Verifica el estado de la API y sus dependencias.

**Respuesta ejemplo:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "environment": {
    "python_version": "3.10.5",
    "working_directory": "/home/deka/mysite",
    "files_exist": {
      "model": true,
      "vectorizer": true
    }
  }
}
```

### Predicción de Sentimiento
```
POST /predict
```
Analiza el sentimiento de un texto.

**Parámetros del Body:**
```json
{
  "review": "Texto del review a analizar"
}
```

**Respuesta ejemplo:**
```json
{
  "sentiment": "Positive",
  "confidence": 0.85,
  "processed_text": "texto procesado"
}
```

## Uso con cURL

### Verificar estado:
```bash
curl -X GET http://deka.pythonanywhere.com/health
```

### Realizar predicción:
```bash
curl -X POST http://deka.pythonanywhere.com/predict \
     -H "Content-Type: application/json" \
     -d '{"review": "This app is amazing!"}'
```

## Uso con Python

```python
import requests

def predict_sentiment(review):
    response = requests.post(
        "http://deka.pythonanywhere.com/predict",
        json={"review": review}
    )
    return response.json()

# Ejemplo de uso
result = predict_sentiment("This app is amazing!")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']}")
```

## Códigos de Estado

- 200: Respuesta exitosa
- 400: Petición incorrecta (ej: review faltante)
- 500: Error interno del servidor
- 503: Servicio no disponible (modelos no cargados)
