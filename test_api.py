#!/usr/bin/env python3
"""
Script de prueba para la API RAG con Ollama
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_root():
    """Prueba el endpoint raíz"""
    print("🔹 Probando endpoint raíz (GET /)...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    print()

def test_health():
    """Prueba el endpoint de salud"""
    print("🔹 Probando endpoint de salud (GET /health)...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    print()

def test_models():
    """Prueba el endpoint de modelos"""
    print("🔹 Probando endpoint de modelos (GET /models)...")
    response = requests.get(f"{BASE_URL}/models")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    print()

def test_ask(question: str):
    """Prueba el endpoint de preguntas"""
    print(f"🔹 Probando endpoint de preguntas (POST /ask)...")
    print(f"Pregunta: {question}")
    
    payload = {
        "question": question,
        "top_k": 4,
        "model": "llama3.2:1b"
    }
    
    response = requests.post(f"{BASE_URL}/ask", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n📝 Respuesta:")
        print(f"  {data['answer']}")
        print(f"\n📊 Metadata:")
        print(f"  - Modelo usado: {data['model_used']}")
        print(f"  - Modelo embeddings: {data['embedding_model']}")
        print(f"  - Fuentes encontradas: {data['sources_found']}")
    else:
        print(f"Error: {response.text}")
    print()

if __name__ == "__main__":
    print("=" * 70)
    print("🦙 PRUEBA DE API RAG CON OLLAMA (100% GRATIS)")
    print("=" * 70)
    print()
    
    # Pruebas
    test_root()
    test_health()
    test_models()
    
    # Preguntas de prueba
    preguntas = [
        "¿Qué es RAG?",
        "¿Cuáles son las ventajas del RAG?",
        "¿Qué tecnologías usa este proyecto?"
    ]
    
    for pregunta in preguntas:
        test_ask(pregunta)
    
    print("=" * 70)
    print("✅ Pruebas completadas")
    print("=" * 70)
