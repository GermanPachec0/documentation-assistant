"""Única puerta de entrada al LLM.

Todo el proyecto llama a `generar(...)` y nunca a google-genai directamente.
Si mañana se cambia a Ollama u otro proveedor, se toca solo este archivo.
"""

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

# El modelo se elige en .env (GEMINI_MODEL=...) para poder cambiarlo sin tocar código.
# Elegilo de la lista que te da client.models.list() (Paso 0).
MODELO = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Temperatura baja: queremos respuestas fieles al contexto y repetibles, no creativas.
TEMPERATURA = 0.1

# Errores que vale la pena reintentar: 429 = límite del plan gratuito, 503 = servidor ocupado.
_REINTENTABLES = {429, 500, 503}
_MAX_INTENTOS = 4

_cliente: genai.Client | None = None


def _get_cliente() -> genai.Client:
    # Se crea una sola vez y se reutiliza (lee GEMINI_API_KEY del entorno).
    global _cliente
    if _cliente is None:
        _cliente = genai.Client()
    return _cliente


def generar(prompt: str, sistema: str | None = None, temperatura: float = TEMPERATURA) -> dict:
    """Manda el prompt al LLM y devuelve el texto más el uso de tokens.

    - `sistema`: instrucciones fijas (reglas de comportamiento), separadas de
      los datos variables (contexto + pregunta) que van en `prompt`.
    - Devuelve un dict para que quien llama pueda registrar costo y latencia.
    """
    config = types.GenerateContentConfig(temperature=temperatura, system_instruction=sistema)

    for intento in range(1, _MAX_INTENTOS + 1):
        inicio = time.perf_counter()
        try:
            respuesta = _get_cliente().models.generate_content(
                model=MODELO, contents=prompt, config=config
            )
            break
        except errors.APIError as e:
            if e.code not in _REINTENTABLES or intento == _MAX_INTENTOS:
                raise
            espera = 2**intento  # 2, 4, 8 segundos: "backoff exponencial"
            print(f"[llm] error {e.code}, reintento {intento} en {espera}s")
            time.sleep(espera)

    uso = respuesta.usage_metadata
    return {
        "texto": respuesta.text or "",
        "modelo": MODELO,
        "tokens_entrada": uso.prompt_token_count if uso else None,
        "tokens_salida": uso.candidates_token_count if uso else None,
        "tokens_razonamiento": uso.thoughts_token_count if uso else None,
        "segundos": round(time.perf_counter() - inicio, 2),
    }


if __name__ == "__main__":
    r = generar("Respondé solo con la palabra: funciona")
    print(r)
