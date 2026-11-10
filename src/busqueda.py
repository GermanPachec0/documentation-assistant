"""Estrategias de búsqueda. Todas con la MISMA firma:

    buscar(pregunta: str, k: int) -> list[str]   # padre_ids, ordenados, sin repetidos

Las dos indexan HIJOS (las mismas unidades, con los mismos ids que Chroma) y
devuelven PADRES, así se pueden comparar y combinar (RRF, Paso 5).
"""

import re
import unicodedata
from functools import cache

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from ingest import MODELO, PREFIJO_PREGUNTA, abrir_coleccion


# --- recursos compartidos: se cargan UNA vez, la primera que se usan ---


@cache
def _modelo() -> SentenceTransformer:
    return SentenceTransformer(MODELO)


@cache
def _coleccion():
    return abrir_coleccion()


def _padres_en_orden(padre_ids: list[str], k: int) -> list[str]:
    # De hijo a padre, deduplicando y CONSERVANDO el orden.
    return list(dict.fromkeys(padre_ids))[:k]


# --- semántica ---


def buscar_semantica(pregunta: str, k: int) -> list[str]:
    vector = _modelo().encode([PREFIJO_PREGUNTA + pregunta], normalize_embeddings=True)
    coleccion = _coleccion()

    # Si se piden exactamente k hijos, pueden venir 5 del mismo padre
    # (insert-cuenta-comitente tiene 20) y quedar 1-2 padres distintos.
    # Se piden más, y se duplica hasta tener k padres o agotar la colección.
    total = coleccion.count()
    n = min(k * 4, total)
    while True:
        res = coleccion.query(query_embeddings=vector.tolist(), n_results=n, include=["metadatas"])
        padres = _padres_en_orden([m["padre_id"] for m in res["metadatas"][0]], k)
        if len(padres) >= k or n >= total:
            return padres
        n = min(n * 2, total)


# --- BM25 (palabras exactas) ---

# Palabras que aparecen en casi todo y no distinguen nada.
_STOPWORDS = {
    "a", "al", "como", "con", "de", "del", "el", "en", "es", "la", "las", "lo",
    "los", "para", "por", "que", "se", "si", "su", "un", "una", "y", "o",
    "the", "of", "to", "is", "what", "how", "a", "an",
}
# Un "nombre" con guiones (get-tipos-contrib-retencion) o una palabra suelta.
_PALABRA = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenizar(texto: str) -> list[str]:
    """Minúsculas, sin tildes, sin stopwords.

    Los nombres con guiones se guardan ENTEROS y además partidos:
    "get-tipos-contrib-retencion" → ese token + "get", "tipos", "contrib", "retencion".
    Así matchea tanto quien escribe el nombre del endpoint como quien escribe
    "tipos de retención". Sin stemming: "actualizo" ≠ "actualiza" (a propósito,
    es la versión más simple; se mide primero).
    """
    sin_tildes = unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode()
    tokens: list[str] = []
    for palabra in _PALABRA.findall(sin_tildes):
        partes = palabra.split("-")
        if len(partes) > 1:
            tokens.append(palabra)
        tokens += partes
    return [t for t in tokens if t not in _STOPWORDS]


@cache
def _indice_bm25() -> tuple[BM25Okapi, list[str]]:
    # Mismos textos y mismos ids que Chroma: se leen de la colección.
    # Solo los hijos: los encabezados son un vector extra para la búsqueda
    # semántica (así su efecto se mide aislado; para BM25 serían texto repetido).
    corpus = _coleccion().get(where={"unidad": "hijo"}, include=["documents", "metadatas"])
    bm25 = BM25Okapi([tokenizar(doc) for doc in corpus["documents"]])
    padre_ids = [m["padre_id"] for m in corpus["metadatas"]]
    return bm25, padre_ids


def buscar_bm25(pregunta: str, k: int) -> list[str]:
    bm25, padre_ids = _indice_bm25()
    puntajes = bm25.get_scores(tokenizar(pregunta))  # uno por hijo
    orden = sorted(range(len(puntajes)), key=lambda i: puntajes[i], reverse=True)
    return _padres_en_orden([padre_ids[i] for i in orden], k)


# --- híbrida: RRF (Reciprocal Rank Fusion) ---

# Constante del paper original. Suaviza la diferencia entre puestos: sin ella,
# 1/1 vs. 1/2 haría que el 1.º de cualquier ranking aplaste al resto.
K_RRF = 60
# Cuántos candidatos se le piden a CADA búsqueda. Más que k: si un padre está
# 7.º en una y 1.º en la otra, tiene que poder sumar los dos términos.
# Medido en eval/evaluar.py (barrido de N).
N_CANDIDATOS = 20


def fusionar_rrf(rankings: list[list[str]], k: int, k_rrf: int = K_RRF) -> list[str]:
    """puntaje(padre) = Σ 1 / (k_rrf + posición), posiciones desde 1.

    Usa solo POSICIONES, no puntajes: la semántica da cosenos (0.8-0.9) y
    BM25 da números sin techo; sumarlos directamente no tendría sentido.
    """
    puntajes: dict[str, float] = {}
    for ranking in rankings:
        for posicion, padre in enumerate(ranking, start=1):
            puntajes[padre] = puntajes.get(padre, 0.0) + 1 / (k_rrf + posicion)
    # sorted es estable: ante empate queda el que apareció primero.
    return sorted(puntajes, key=puntajes.__getitem__, reverse=True)[:k]


def buscar_hibrida(
    pregunta: str, k: int, n_candidatos: int = N_CANDIDATOS, k_rrf: int = K_RRF
) -> list[str]:
    n = max(n_candidatos, k)
    return fusionar_rrf([buscar_semantica(pregunta, n), buscar_bm25(pregunta, n)], k, k_rrf)
