"""Evaluación de la RECUPERACIÓN (no de la respuesta del LLM).

Uso: PYTHONPATH=src uv run python eval/evaluar.py

Toda estrategia de búsqueda tiene la MISMA firma:
    buscar(pregunta: str, k: int) -> list[str]   # padre_ids, ordenados, sin repetidos
Así, en el Paso 5, BM25, RRF o "agrupar por endpoint" se enchufan sin tocar evaluar().
"""

from collections.abc import Callable
from pathlib import Path

import yaml

from busqueda import buscar_bm25, buscar_hibrida, buscar_semantica

RUTA_PREGUNTAS = Path("eval/preguntas.yaml")

Buscador = Callable[[str, int], list[str]]


def cargar_preguntas(ruta: Path = RUTA_PREGUNTAS) -> list[dict]:
    return yaml.safe_load(ruta.read_text(encoding="utf-8"))["preguntas"]


def evaluar(buscar: Buscador, preguntas: list[dict], k: int = 5) -> dict[str, float]:
    con_respuesta = [p for p in preguntas if not p.get("sin_respuesta")]
    sin_respuesta = [p for p in preguntas if p.get("sin_respuesta")]

    filas = []  # (pregunta, posición del primer esperado o None, resultados)
    for p in con_respuesta:
        resultados = buscar(p["pregunta"], k)
        posicion = next(
            (i for i, padre in enumerate(resultados, start=1) if padre in p["esperados"]),
            None,
        )
        filas.append((p, posicion, resultados))

    n = len(filas)
    hit1 = sum(pos == 1 for _, pos, _ in filas) / n
    hitk = sum(pos is not None for _, pos, _ in filas) / n
    mrr = sum(1 / pos for _, pos, _ in filas if pos) / n

    print(f"\nPreguntas con respuesta: {n}   (sin respuesta, no se evalúan acá: {len(sin_respuesta)})")
    print(f"  hit@1 = {hit1:.2f}   hit@{k} = {hitk:.2f}   MRR@{k} = {mrr:.2f}")

    print(f"\n{'id':4} {'pos':>4}  {'categoría':13} pregunta")
    for p, pos, _ in filas:
        print(f"{p['id']:4} {pos if pos else '—':>4}  {p['categoria']:13} {p['pregunta']}")

    fallas = [(p, r) for p, pos, r in filas if pos is None]
    if fallas:
        print(f"\nFallan en el top {k} (qué trajo en su lugar, top 3):")
        for p, resultados in fallas:
            print(f"  {p['id']} {p['pregunta']}")
            print(f"       esperado: {', '.join(p['esperados'])}")
            for r in resultados[:3]:
                print(f"       trajo:    {r}")

    return {"hit@1": hit1, f"hit@{k}": hitk, "mrr": mrr}


if __name__ == "__main__":
    import sys

    estrategias = {
        "semantica": buscar_semantica,
        "bm25": buscar_bm25,
        "hibrida": buscar_hibrida,
    }
    # Uso: evaluar.py [semantica|bm25 ...]   (sin argumentos: todas)
    elegidas = sys.argv[1:] or list(estrategias)
    preguntas = cargar_preguntas()
    resumen = {}
    for nombre in elegidas:
        print(f"\n{'=' * 70}\n{nombre}\n{'=' * 70}")
        resumen[nombre] = evaluar(estrategias[nombre], preguntas)

    if len(resumen) > 1:
        print(f"\n{'=' * 70}\nResumen")
        for nombre, m in resumen.items():
            print(f"  {nombre:10} " + "   ".join(f"{k} = {v:.2f}" for k, v in m.items()))
