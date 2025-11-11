# Guía de Uso - API RAG con Ollama

## Servidor de Aplicación

El servidor se encuentra disponible en: http://localhost:8000

## Endpoints Disponibles

### 1. Documentación Interactiva (Swagger UI)

Acceder a la documentación interactiva de la API mediante:
```
http://localhost:8000/docs
```

La interfaz Swagger UI permite visualizar todos los endpoints disponibles, sus esquemas de datos y ejecutar peticiones de prueba directamente desde el navegador.

### 2. Endpoint de Consulta (POST /ask)

Este endpoint permite realizar consultas sobre el corpus de documentos indexados.

**Ejemplo usando curl:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Qué es RAG?",
    "top_k": 4,
    "model": "llama3.2:1b"
  }'
```

**Ejemplo usando Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={
        "question": "¿Qué es RAG?",
        "top_k": 4,
        "model": "llama3.2:1b"
    }
)

print(response.json()["answer"])
```

**Ejemplo usando JavaScript (Fetch API):**
```javascript
fetch('http://localhost:8000/ask', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    question: '¿Qué es RAG?',
    top_k: 4,
    model: 'llama3.2:1b'
  })
})
.then(r => r.json())
.then(data => console.log(data.answer));
```

### 3. Verificación de Estado del Servicio (GET /health)

Endpoint para verificar el estado operacional del servidor y sus dependencias.

```bash
curl http://localhost:8000/health
```

### 4. Listado de Modelos Disponibles (GET /models)

Consulta los modelos de lenguaje disponibles en el sistema.

```bash
curl http://localhost:8000/models
```

### 5. Proceso de Ingesta de Documentos (POST /ingest)

Inicia el proceso de indexación de documentos del directorio `data/`.

```bash
curl -X POST http://localhost:8000/ingest
```

## Operación del Servidor

### Iniciar el Servidor de Aplicación
```bash
source .venv/bin/activate
uvicorn src.app_ollama:app --reload --port 8000
```

### Detener el Servidor

Para detener el servidor, ejecutar la combinación de teclas `Ctrl+C` en la terminal activa.

### Visualización de Logs

Los registros de operación se muestran en tiempo real en la terminal donde se ejecutó el comando uvicorn.

### Adición de Nuevos Documentos al Corpus

Para indexar documentos adicionales:

1. Colocar los archivos (PDF, TXT, MD) en el directorio `data/`
2. Ejecutar el módulo de ingesta:
   ```bash
   python -m src.ingest_ollama
   ```
   Alternativamente, utilizar el endpoint HTTP: `POST /ingest`

## Formato de Respuesta

### Estructura de Respuesta JSON

```json
{
  "answer": "El RAG es una técnica que combina...",
  "model_used": "llama3.2:1b",
  "embedding_model": "nomic-embed-text",
  "sources_found": 4
}
```

## Integración con Aplicaciones Frontend

### Ejemplo de Integración con React
```jsx
async function askQuestion(question) {
  const response = await fetch('http://localhost:8000/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question })
  });
  const data = await response.json();
  return data.answer;
}
```

## Resolución de Problemas Comunes

### Error: Servicio Ollama No Disponible
```bash
# Verificar que el servicio Ollama esté en ejecución
brew services start ollama

# Alternativamente, iniciar manualmente
ollama serve
```

### Error: Modelo No Encontrado

```bash
# Descargar los modelos requeridos
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

### Puerto 8000 en Uso

```bash
# Especificar puerto alternativo
uvicorn src.app_ollama:app --reload --port 8001
```

## Recomendaciones de Implementación

### Consideraciones de Desarrollo

1. **Documentación Interactiva**: Utilizar la interfaz Swagger UI disponible en `/docs` para pruebas y exploración de la API.

2. **Selección de Modelos**: Para aplicaciones que requieren mayor precisión semántica, considerar el uso de `llama3.2:3b` en lugar del modelo ligero `:1b`.

3. **Configuración CORS**: Para aplicaciones web que requieran acceso desde dominios externos, es necesario habilitar CORS.

4. **Entorno de Producción**: En ambientes productivos, eliminar la bandera `--reload` y considerar el uso de servidores ASGI como Gunicorn con workers de Uvicorn.

## Configuración de CORS

Para habilitar Cross-Origin Resource Sharing (CORS) en el archivo `app_ollama.py`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Nota de Seguridad**: En entornos de producción, reemplazar `allow_origins=["*"]` con una lista explícita de dominios autorizados.

## Estado del Sistema

### Resumen de Funcionalidades Activas

- Servidor API disponible en http://localhost:8000
- Documentación interactiva accesible en http://localhost:8000/docs
- Sistema de embeddings y generación local mediante Ollama (sin dependencias de servicios externos de pago)
- Procesamiento de datos completamente local, garantizando privacidad de la información
