# Imagen para Hugging Face Spaces (sdk: docker). También corre igual en cualquier
# servidor con Docker:  docker build -t asistente . && docker run -p 7860:7860 -e GEMINI_API_KEY=... asistente

FROM python:3.12-slim

# uv, la misma herramienta que se usa en desarrollo (versiones fijadas en uv.lock)
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

# Hugging Face Spaces corre los contenedores con el usuario 1000: se crea y se usa ese.
# La carpeta de la app se crea acá, con su dueño: WORKDIR la crearía como root
# y el usuario 1000 no podría escribir el .venv adentro.
RUN useradd -m -u 1000 user && mkdir -p /home/user/app && chown user:user /home/user/app
USER user

ENV HOME=/home/user \
    PATH=/home/user/app/.venv/bin:$PATH \
    PYTHONPATH=/home/user/app/src \
    HF_HOME=/home/user/.cache/huggingface \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1
# UV_NO_CACHE: sin esto, uv guarda una copia de cada paquete descargado
# (~1.7 GB) que en la imagen no sirve para nada.
WORKDIR /home/user/app

# 1) Dependencias primero: Docker cachea esta capa y no reinstala todo
#    cada vez que cambia una línea de código. En Linux, PyTorch va en versión CPU.
COPY --chown=user pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev

# 2) El código
COPY --chown=user . .

# 3) Datos, al construir la imagen (no al arrancar): descarga el PDF público,
#    lo fragmenta, baja el modelo de embeddings e indexa en Chroma.
#    Así la app arranca rápido y no depende de la URL del PDF en cada reinicio.
RUN python src/preparar.py

EXPOSE 7860
CMD ["streamlit", "run", "src/app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
