"""Prepara todo lo necesario para que la app responda, desde cero.

Descarga el PDF público de ESCO (si no está), lo convierte en fragmentos y los
indexa en Chroma. Es lo que corre el servidor al construirse: el repo no
incluye ni el PDF ni texto extraído de él, solo el código que lo procesa.

Uso: PYTHONPATH=src uv run python src/preparar.py
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

from ingest import ingestar
from readers.pdf_esco import guardar_json, leer_pdf

URL_PDF = "https://escoapi.primary.com.ar/swagger-ui/documentos/Instructivo_Esco_Api_escobolsa_v9.pdf"
RUTA_PDF = Path("data/raw/Instructivo_Esco_Api_escobolsa_v9.pdf")
RUTA_CHUNKS = Path("data/chunks/esco.json")

# Huella del PDF con el que se construyó y verificó el lector (lista DESCARTAR,
# umbrales medidos, colores). Si Primary publica otra versión con el mismo
# nombre, el lector podría fallar en silencio: mejor detenerse y avisar.
SHA256_ESPERADO = "d6b6b706c070bfe35b4aea6b6e2e24e13d618786116cce183a3469fa035332bb"


def descargar_pdf() -> None:
    if RUTA_PDF.exists():
        print(f"PDF ya presente: {RUTA_PDF}")
        return
    RUTA_PDF.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {URL_PDF} …")
    urllib.request.urlretrieve(URL_PDF, RUTA_PDF)


def verificar_pdf() -> None:
    huella = hashlib.sha256(RUTA_PDF.read_bytes()).hexdigest()
    if huella != SHA256_ESPERADO:
        sys.exit(
            f"El PDF descargado no es la versión esperada (sha256 {huella[:12]}…).\n"
            "Puede ser una versión nueva de la documentación: revisá el lector "
            "(títulos, DESCARTAR, umbrales) antes de actualizar SHA256_ESPERADO."
        )


def main() -> None:
    descargar_pdf()
    verificar_pdf()
    fragmentos = leer_pdf(RUTA_PDF)
    guardar_json(fragmentos, RUTA_CHUNKS)
    print(f"{len(fragmentos)} fragmentos → {RUTA_CHUNKS}")
    coleccion, _ = ingestar()
    print(f"Índice listo: {coleccion.count()} vectores")


if __name__ == "__main__":
    main()
