# Guía de Implementación: Ollama para Procesamiento Local

Esta guía detalla la configuración del proyecto RAG utilizando Ollama como alternativa a servicios de API comerciales.

## Descripción de Ollama

Ollama es un framework que permite la ejecución de modelos de lenguaje grandes (LLMs) de manera local, eliminando la dependencia de servicios externos de pago y garantizando la privacidad de los datos procesados.

## Proceso de Instalación

### 1. Instalación del Software Base

```bash
# Método 1: Descarga directa desde el sitio oficial
# Disponible en: https://ollama.ai/download

# Método 2: Instalación mediante Homebrew (macOS)
brew install ollama
```

### 2. Inicialización del Servicio

```bash
ollama serve
```

El servicio quedará activo en `http://localhost:11434`. Se recomienda mantener esta terminal en ejecución durante la operación del sistema.

### 3. Descarga de Modelos de Lenguaje

Ejecutar los siguientes comandos en una terminal separada:

```bash
# Modelo para generación de embeddings (transformación de texto a vectores)
ollama pull nomic-embed-text

# Modelo para generación de respuestas conversacionales
ollama pull llama3.2
```

**Modelos alternativos** (para sistemas con recursos limitados):
```bash
ollama pull llama3.2:1b      # Versión optimizada (1B parámetros)
ollama pull qwen2.5:3b        # Alternativa eficiente (3B parámetros)
ollama pull phi3:mini         # Modelo compacto de Microsoft
```

## Configuración del Proyecto

### 1. Instalación de Dependencias

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Proceso de Ingesta Documental

```bash
python -m src.ingest_ollama
```

Este proceso realiza las siguientes operaciones:
- Lectura de documentos del directorio `data/`
- Generación de embeddings mediante `nomic-embed-text` (procesamiento local)
- Indexación en la base de datos vectorial Pinecone

### 3. Sistema de Consultas

```bash
python -m src.query_ollama
```

El sistema utilizará:
- Embeddings locales para la recuperación de información
- Modelo `llama3.2` local para la generación de respuestas

## Análisis Comparativo: Servicios Comerciales vs. Procesamiento Local

| Característica | Servicios Comerciales (OpenAI) | Ollama (Local) |
|----------------|-------------------------------|----------------|
| **Modelo de Costos** | Pago por uso (por token) | Sin costo de operación |
| **Privacidad de Datos** | Transmisión a servidores externos | Procesamiento completamente local |
| **Latencia** | Dependiente de conectividad | Determinada por hardware local |
| **Calidad de Respuestas** | Consistentemente alta | Variable según modelo seleccionado |
| **Requisitos Técnicos** | Conexión a internet + credenciales API | RAM mínima de 8GB (recomendado 16GB+) |

## Resolución de Problemas

### Error: Conexión Rechazada
```bash
# Verificar que el servicio Ollama esté activo
ollama serve
```

### Error: Modelo No Encontrado

```bash
# Descargar modelos requeridos
ollama pull nomic-embed-text
ollama pull llama3.2
```

### Rendimiento Degradado

Para sistemas con recursos limitados, utilizar modelos optimizados:

```bash
# Descargar modelo ligero
ollama pull llama3.2:1b
```

Posteriormente, modificar el archivo `src/query_ollama.py` en la línea 28:
```python
llm = OllamaLLM(model="llama3.2:1b", temperature=0)
```

## Ventajas del Procesamiento Local

### Beneficios Principales

1. **Eliminación de costos operacionales**: No existen cargos por uso de tokens o llamadas a API.

2. **Garantía de privacidad**: Los datos procesados permanecen en el sistema local sin transmisión externa.

3. **Ausencia de limitaciones de tasa**: No existen restricciones de rate limiting o cuotas de uso.

4. **Independencia de conectividad**: Operación completamente funcional sin conexión a internet (posterior a la descarga inicial de modelos).

## Alternativa: Uso de Servicios Comerciales

Para retornar al uso de servicios comerciales (OpenAI):
```bash
python -m src.ingest    # Implementación con OpenAI
python -m src.query     # Implementación con OpenAI
```

**Nota**: Requiere configuración previa de credenciales API y disponibilidad de créditos en la cuenta de OpenAI.
