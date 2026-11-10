---
title: Asistente de documentación de APIs (ESCO)
emoji: 📘
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Asistente de documentación de APIs con RAG

Responde preguntas sobre la documentación de **EscoAPI v9** (back office bursátil de
Primary): devuelve el endpoint correcto, sus parámetros, un request de ejemplo con
datos ficticios y cita la página del PDF de donde sale cada dato.

Construido a mano, pieza por pieza, para aprender cómo funciona un RAG: lector de PDF
propio, chunking por endpoint, embeddings locales, búsqueda semántica + BM25 con RRF,
y generación con citas verificadas por código.

> README en construcción: la versión completa (arquitectura, métricas, decisiones y
> limitaciones) llega en el Paso 9.

## Correr localmente

```bash
uv sync
cp .env.example .env            # y completá GEMINI_API_KEY
PYTHONPATH=src uv run python src/preparar.py     # descarga el PDF, fragmenta e indexa
PYTHONPATH=src uv run streamlit run src/app.py
```

## Fuente

Documentación pública de EscoAPI v9:
<https://escoapi.primary.com.ar/swagger-ui/documentos/Instructivo_Esco_Api_escobolsa_v9.pdf>.
El repositorio no incluye el PDF ni texto extraído de él: se descargan y procesan al
construir la app.

Los ejemplos de request son ilustrativos y **no fueron probados contra la API real**.
