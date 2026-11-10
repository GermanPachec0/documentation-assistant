"""Interfaz de chat del asistente de documentación.

Uso (desde la raíz del proyecto):
    PYTHONPATH=src uv run streamlit run src/app.py

Cómo funciona Streamlit, en una línea: cada vez que el usuario hace algo
(escribe, aprieta un botón), el script ENTERO se vuelve a ejecutar de arriba a
abajo. Lo que tiene que sobrevivir entre ejecuciones (el historial del chat)
se guarda en `st.session_state`.
"""

import json
import os
import re
import threading
import time
from collections import deque

from pathlib import Path

import streamlit as st

# Las rutas del proyecto ("data/chroma", "data/chunks"…) son relativas a la raíz
# del repo. Así funcionan sin importar desde qué carpeta se lance la app
# (en Streamlit Community Cloud no lo elegimos nosotros).
os.chdir(Path(__file__).resolve().parent.parent)

import llm  # noqa: E402  (después del chdir, a propósito)
from generation import responder  # noqa: E402
from ingest import MODELO as MODELO_EMBEDDINGS  # noqa: E402
from ingest import abrir_coleccion  # noqa: E402


@st.cache_resource(show_spinner="Preparando la documentación (solo la primera vez, unos 2 minutos)…")
def asegurar_indice() -> int:
    """Si el índice no existe (servidor recién creado), lo arma desde cero.

    En Docker esto pasa al construir la imagen; en Streamlit Community Cloud no
    hay paso de construcción, así que lo hace la app en su primer arranque.
    cache_resource: corre una sola vez por proceso, no en cada clic.
    """
    if abrir_coleccion().count() == 0:
        import preparar

        preparar.main()
    return abrir_coleccion().count()


asegurar_indice()

# --- Configuración -----------------------------------------------------------

SISTEMAS = {"ESCO (EscoAPI v9)": "esco"}  # una entrada por documentación indexada

# Límites de uso: la app pública gasta la cuota de Gemini de quien la despliega.
# - por sesión: frena a una persona (pero se esquiva recargando la página);
# - global por hora: frena el total, sumando a todos los visitantes.
# Se pueden cambiar con variables de entorno sin tocar el código.
MAX_POR_SESION = int(os.getenv("MAX_PREGUNTAS_POR_SESION", "15"))
MAX_POR_HORA = int(os.getenv("MAX_PREGUNTAS_POR_HORA", "120"))


@st.cache_resource
def _registro_global() -> tuple[deque, threading.Lock]:
    # cache_resource: UN solo objeto compartido por todas las sesiones del proceso.
    return deque(), threading.Lock()


def hay_cupo_global() -> bool:
    """Registra la pregunta si entra en el tope por hora; False si no hay cupo."""
    tiempos, lock = _registro_global()
    ahora = time.time()
    with lock:
        while tiempos and ahora - tiempos[0] > 3600:
            tiempos.popleft()  # se olvidan las preguntas de hace más de una hora
        if len(tiempos) >= MAX_POR_HORA:
            return False
        tiempos.append(ahora)
        return True

# Preguntas listas para probar, agrupadas por lo que ponen a prueba.
EJEMPLOS = {
    "Encontrar el endpoint correcto": [
        "¿Cómo actualizo el domicilio de un comitente?",
        "¿Cómo creo una persona jurídica?",
        "¿Cómo anulo una orden?",
        "¿Cómo puedo agregarle saldo a una cuenta comitente?",
    ],
    "Nombres exactos": [
        "¿Qué endpoint devuelve porcPartAccionaria?",
        "¿Qué devuelve get-tipos-contrib-retencion?",
    ],
    "Reglas de negocio": [
        "¿Qué FCI puedo usar para operar?",
    ],
    "Casos difíciles": [
        "¿Qué parámetros recibe update-condominio?",
        "¿Cómo puedo obtener las operaciones de un cliente?",
        "How do I create a new client account?",
        "Dame la receta del pan casero",
    ],
}

st.set_page_config(page_title="Asistente ESCO", page_icon="📘", layout="wide")

# --- Barra lateral: configuración y modelos ----------------------------------

pregunta_ejemplo = None  # se completa si el usuario aprieta una pregunta de ejemplo

with st.sidebar:
    st.header("Preguntas de ejemplo")
    st.caption("Hacé clic en una para preguntarla.")
    for grupo, preguntas in EJEMPLOS.items():
        st.markdown(f"**{grupo}**")
        for p in preguntas:
            # key única: el texto de la pregunta (Streamlit necesita distinguir los botones)
            if st.button(p, key=f"ej-{p}", use_container_width=True):
                pregunta_ejemplo = p

    st.divider()
    st.header("Configuración")
    sistema = st.selectbox(
        "Documentación",
        list(SISTEMAS),
        help="Por ahora hay una sola fuente indexada. Con más fuentes, la búsqueda se filtra por la elegida.",
    )
    k = st.slider("Fragmentos que recibe el LLM (k)", 1, 10, 5)

    st.divider()
    st.subheader("Modelos")
    st.markdown(f"**LLM:** `{llm.MODELO}`  \n**Embeddings:** `{MODELO_EMBEDDINGS}`")

    st.divider()
    if st.button("Borrar conversación", use_container_width=True):
        st.session_state.historial = []
        st.rerun()

    st.caption(
        "Los ejemplos de request salen de la documentación y no fueron "
        "probados contra la API real."
    )
    st.caption(f"Preguntas usadas en esta sesión: {st.session_state.get('preguntas_hechas', 0)} de {MAX_POR_SESION}")

# --- Estado: historial del chat ----------------------------------------------

if "historial" not in st.session_state:
    st.session_state.historial = []  # lista de dicts: {"rol": ..., "texto": ..., "resultado": ...}
if "preguntas_hechas" not in st.session_state:
    st.session_state.preguntas_hechas = 0


_BLOQUE_JSON = re.compile(r"```json\s*\n(.*?)```", re.DOTALL)


def verificar_json(texto: str) -> None:
    """El LLM genera, el código verifica: avisa si un bloque ```json no es JSON válido.

    El PDF tiene ejemplos rotos (comas finales, comillas tipográficas “DNI”) y
    el modelo puede copiarlos tal cual. Mejor avisarle al usuario que dejar
    que lo descubra cuando la API le devuelva un error.
    """
    for n, bloque in enumerate(_BLOQUE_JSON.findall(texto), start=1):
        contenido = bloque.strip()
        # Un JSON siempre empieza con { o [. Si no, el modelo etiquetó como
        # ```json algo que no lo es (típico: la ruta de un GET con query string).
        if not contenido.startswith(("{", "[")):
            primera = contenido.splitlines()[0] if contenido else "(vacío)"
            st.caption(f"ℹ️ El bloque {n} está marcado como JSON pero no lo es: `{primera[:80]}`")
            continue
        try:
            json.loads(contenido)
            st.caption(f"✅ El JSON del bloque {n} es válido (sintaxis; no fue probado contra la API).")
        except json.JSONDecodeError as e:
            linea = contenido.splitlines()[e.lineno - 1].strip() if e.lineno else ""
            st.warning(
                f"⚠️ El JSON del bloque {n} no es válido: {e.msg} (línea {e.lineno}: `{linea[:80]}`). "
                "Revisalo antes de usarlo."
            )


def mostrar_resultado(r: dict) -> None:
    """El cuerpo de la respuesta, una línea con modelo/tokens y las fuentes."""
    st.markdown(r["cuerpo"])
    verificar_json(r["cuerpo"])
    st.caption(
        f"🤖 {r['modelo']} · {r['tokens_entrada']:,} tokens de entrada / "
        f"{r['tokens_salida']:,} de salida · {r['segundos']} s"
    )

    citadas = [c for c in r["contexto"] if c["citada"]]
    no_citadas = [c for c in r["contexto"] if not c["citada"]]

    if citadas:
        st.markdown("**Fuentes citadas**")
        for c in citadas:
            with st.expander(f"[{c['n']}] {c['etiqueta']}"):
                st.code(c["texto"], language=None)
    else:
        st.caption("La respuesta no cita ninguna fuente.")

    # Lo que el modelo recibió pero no usó: sirve para depurar la búsqueda.
    if no_citadas:
        with st.expander(f"Recuperadas pero no citadas ({len(no_citadas)})"):
            for c in no_citadas:
                st.markdown(f"[{c['n']}] {c['etiqueta']}")


# --- Pantalla principal ------------------------------------------------------

st.title("📘 Asistente de documentación de APIs")
st.caption(
    "Respondo solo con la documentación indexada y cito la página del PDF. "
    "Escribí tu pregunta abajo o elegí un ejemplo en la barra de la izquierda."
)

# Historial (se redibuja entero en cada ejecución del script)
for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["rol"]):
        if mensaje["rol"] == "user":
            st.markdown(mensaje["texto"])
        else:
            mostrar_resultado(mensaje["resultado"])

# Entrada: lo escrito en el chat o el ejemplo elegido
pregunta = st.chat_input("Preguntá sobre la API…") or pregunta_ejemplo

if pregunta and st.session_state.preguntas_hechas >= MAX_POR_SESION:
    st.warning(
        f"Llegaste al máximo de {MAX_POR_SESION} preguntas de esta demo. "
        "¡Gracias por probarla! El código está en el repositorio del proyecto."
    )
    pregunta = None
elif pregunta and not hay_cupo_global():
    st.warning("La demo recibió muchas preguntas en la última hora. Probá de nuevo en un rato.")
    pregunta = None

if pregunta:
    st.session_state.preguntas_hechas += 1
    st.session_state.historial.append({"rol": "user", "texto": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en la documentación…"):
            try:
                resultado = responder(pregunta, k=k)
            except Exception as e:  # que un error de la API no tire abajo la app
                st.error(f"No pude responder: {e}")
                st.stop()
        mostrar_resultado(resultado)

    st.session_state.historial.append({"rol": "assistant", "resultado": resultado})
