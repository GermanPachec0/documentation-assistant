"""Partir solo los fragmentos que no entran en el modelo de embeddings.

Chunking "padre-hijo" en su versión más simple:
- el padre es el fragmento completo de data/chunks/esco.json (un endpoint);
- los hijos son sus partes, cada una con el encabezado del padre repetido.
El id de un hijo es "<id del padre>#<n>", así que el padre se recupera con
id.split("#")[0], sin agregar campos al modelo.

Decisión 3 — qué se le manda al LLM cuando la búsqueda trae un hijo:
- Padre de hasta MAX_TOKENS_PADRE_AL_LLM tokens → el PADRE completo.
- Padre más grande → el principio (#1) + el hijo más relevante y sus vecinos.
Por qué: el padre completo responde preguntas que cruzan partes ("¿qué es
obligatorio?", "¿cuál es el flujo completo?"). Empezó en 1.500 tokens por
precaución (costo, lost-in-the-middle), pero cortaba insert-cuenta-comitente
(~7.400 tokens) y el LLM no veía los campos ni las consultas previas que pide.
Ahora: hasta 3.000 tokens, entero siempre; más grande, entero si todavía
entra en el presupuesto de 9.000 para todo el contexto (en orden de
relevancia); si no, recortado.
"""

import re

from models import Fragmento

# 500 y no 512: deja lugar para los tokens especiales del modelo.
MAX_TOKENS = 500
# Decisión 3 (ver docstring): hasta este tamaño se manda el padre entero al LLM.
# Además hay un PRESUPUESTO para todo el contexto: un padre más grande que
# MAX_TOKENS_PADRE_AL_LLM igual va entero si todavía entra en el presupuesto
# (así insert-cuenta-comitente, ~7.400, llega completo cuando hace falta).
# Mandar todo entero sin tope llevaba preguntas simples a 14.000+ tokens y
# Flash-Lite empeoraba: más ruido, no más información útil.
PRESUPUESTO_TOKENS_CONTEXTO = 9000
MAX_TOKENS_PADRE_AL_LLM = 3000
PREFIJO_DOC = "passage: "  # e5 embebe los documentos con este prefijo

# Decisión 2: el encabezado es lo que identifica al endpoint (método +
# subtítulo + descripción), o sea, todo lo anterior al primer marcador del
# cuerpo. Mediana medida: 40 tokens. Si pasa de este tope (máximo medido: 300,
# insert-garantia-orden), se repiten solo las 2 primeras líneas (método +
# subtítulo) y el resto de la descripción pasa a la parte 1.
MAX_TOKENS_ENCABEZADO = 100

# Decisión 1: se corta ANTES de estas líneas, nunca en medio de un parámetro.
#   "public …"           → empieza un parámetro (1.482 en los fragmentos largos)
#   Request / Response   → empieza una tabla
#   Ejemplo(s)           → empieza un ejemplo
_INICIO_UNIDAD = re.compile(r"^(public\b|Request\b|Response\b|Ejemplos?\b)", re.IGNORECASE)
_INICIO_CUERPO = re.compile(r"^(Request\b|Response\b|Campo\b|public\b|Entidad:|Parámetro:)", re.IGNORECASE)
# Para que una parte del medio sepa en qué tabla está: "[Response, continúa]"
_ETIQUETA = re.compile(r"^(Request|Response|Ejemplos?)\b", re.IGNORECASE)
_LINEA_CONTINUA = re.compile(r"^\[(Request|Response|Ejemplos?), continúa\]$")


def contar_tokens(modelo, texto: str) -> int:
    """Tokens que ve el modelo, con el prefijo y los tokens especiales."""
    return len(modelo.tokenizer(PREFIJO_DOC + texto, truncation=False)["input_ids"])


def _separar_encabezado(lineas: list[str], modelo) -> tuple[list[str], list[str]]:
    k = next((i for i, l in enumerate(lineas) if _INICIO_CUERPO.match(l.strip())), len(lineas))
    k = max(k, 1)  # como mínimo, la línea "POST endpoint"
    if contar_tokens(modelo, "\n".join(lineas[:k])) > MAX_TOKENS_ENCABEZADO:
        k = min(2, len(lineas))
    return lineas[:k], lineas[k:]


def _unidades(cuerpo: list[str]) -> list[list[str]]:
    """Agrupa las líneas en unidades que no conviene separar (un parámetro, un título de tabla…)."""
    unidades: list[list[str]] = []
    for linea in cuerpo:
        if not unidades or _INICIO_UNIDAD.match(linea.strip()):
            unidades.append([linea])
        else:
            unidades[-1].append(linea)
    return unidades


SUFIJO_ENCABEZADO = "#encabezado"


def encabezado_de(fragmento: Fragmento, modelo) -> Fragmento:
    """Un fragmento con SOLO el encabezado del padre (método + subtítulo + descripción).

    Se embebe como vector aparte, apuntando al mismo padre: en el Paso 4 vimos
    que con solo el encabezado el PUT de domicilio ganaba (0.877), mientras
    que con el cuerpo entero (listas de parámetros) quedaba 9.º.
    El id termina en "#encabezado", así que id.split("#")[0] sigue siendo el padre.
    """
    encabezado, _ = _separar_encabezado(fragmento.texto.splitlines(), modelo)
    return fragmento.model_copy(
        update={"id": fragmento.id + SUFIJO_ENCABEZADO, "texto": "\n".join(encabezado)}
    )


def partir_largos(fragmento: Fragmento, modelo, max_tokens: int = MAX_TOKENS) -> list[Fragmento]:
    if contar_tokens(modelo, fragmento.texto) <= max_tokens:
        return [fragmento]

    encabezado, cuerpo = _separar_encabezado(fragmento.texto.splitlines(), modelo)
    texto_encabezado = "\n".join(encabezado)

    partes: list[list[str]] = []
    actual: list[str] = []
    etiqueta = None  # última tabla vista: Request / Response / Ejemplo

    def cabe(lineas: list[str]) -> bool:
        return contar_tokens(modelo, texto_encabezado + "\n" + "\n".join(lineas)) <= max_tokens

    def nueva_parte() -> list[str]:
        # Si la parte arranca en medio de una tabla, se lo decimos.
        return [f"[{etiqueta}, continúa]"] if etiqueta else []

    for unidad in _unidades(cuerpo):
        m = _ETIQUETA.match(unidad[0].strip())
        if actual and not cabe(actual + unidad):
            partes.append(actual)
            # Si la unidad que abre la parte ya es "Response:" (o Request/Ejemplo),
            # se identifica sola: no lleva "[… continúa]". Antes se escribía la
            # etiqueta VIEJA ("[Request, continúa]") seguida de "Response:".
            actual = [] if m else nueva_parte()
        if m:
            # Se actualiza ANTES de seguir: si esta unidad no entra y hay que
            # cortarla entre líneas, las partes nuevas están dentro de ella.
            etiqueta = m.group(1).capitalize()
        if cabe(actual + unidad):
            actual += unidad
        else:
            # Una unidad sola no entra (ej. un JSON de ejemplo larguísimo):
            # ahí sí no queda otra que cortar entre líneas.
            for linea in unidad:
                if actual and not cabe(actual + [linea]):
                    partes.append(actual)
                    actual = nueva_parte()
                actual.append(linea)
    if actual:
        partes.append(actual)

    # ¿Por qué "#n"? Si dos partes tuvieran el mismo id, al guardarlas en
    # Chroma la segunda pisaría a la primera.
    return [
        fragmento.model_copy(
            update={
                "id": f"{fragmento.id}#{n}",
                "texto": texto_encabezado + "\n" + "\n".join(parte),
            }
        )
        for n, parte in enumerate(partes, start=1)
    ]


if __name__ == "__main__":
    import json
    import statistics

    from sentence_transformers import SentenceTransformer

    modelo = SentenceTransformer("intfloat/multilingual-e5-small")
    with open("data/chunks/esco.json", encoding="utf-8") as f:
        padres = [Fragmento(**d) for d in json.load(f)]

    hijos = [h for p in padres for h in partir_largos(p, modelo)]
    tokens = [contar_tokens(modelo, h.texto) for h in hijos]
    partidos = [p for p in padres if contar_tokens(modelo, p.texto) > MAX_TOKENS]

    print(f"Padres: {len(padres)}  →  hijos: {len(hijos)}")
    print(f"Padres partidos: {len(partidos)}")
    print(f"Tokens por hijo: mediana {statistics.median(tokens):.0f}, máximo {max(tokens)}")
    assert max(tokens) <= MAX_TOKENS, max(tokens)
    assert len({h.id for h in hijos}) == len(hijos), "ids repetidos"

    # Chequeo ESTRICTO: a cada hijo se le saca el encabezado y la etiqueta
    # "[… continúa]"; los cuerpos concatenados EN ORDEN tienen que ser
    # exactamente el cuerpo del padre. (Comparar conjuntos de líneas no sirve:
    # "}" o "]" aparecen mil veces y una perdida pasaría desapercibida.)
    for p in partidos:
        encabezado, cuerpo_padre = _separar_encabezado(p.texto.splitlines(), modelo)
        reconstruido: list[str] = []
        for h in (h for h in hijos if h.id.split("#")[0] == p.id):
            lineas = h.texto.splitlines()
            assert lineas[: len(encabezado)] == encabezado, (h.id, "encabezado distinto")
            resto = lineas[len(encabezado) :]
            if resto and _LINEA_CONTINUA.match(resto[0]):
                resto = resto[1:]
            reconstruido += resto
        assert reconstruido == cuerpo_padre, p.id
    print("OK: ningún hijo pasa de", MAX_TOKENS, "tokens, ids únicos, y los cuerpos")
    print("    concatenados en orden reconstruyen EXACTAMENTE a cada padre")

    # La etiqueta nunca contradice a la línea que sigue
    for h in hijos:
        lineas = h.texto.splitlines()
        for a, b in zip(lineas, lineas[1:]):
            assert not (_LINEA_CONTINUA.match(a) and _ETIQUETA.match(b.strip())), (h.id, a, b)
    print("OK: ninguna parte dice '[X, continúa]' seguido de Request/Response/Ejemplo")

    # Para la decisión 3: cuántos padres se mandarían enteros al LLM
    chicos = sum(contar_tokens(modelo, p.texto) <= MAX_TOKENS_PADRE_AL_LLM for p in padres)
    print(f"\nPadres de hasta {MAX_TOKENS_PADRE_AL_LLM} tokens (van enteros al LLM): {chicos} de {len(padres)}")

    print("\nPartes de los más grandes:")
    for p in sorted(partidos, key=lambda p: -len(p.texto))[:5]:
        n = sum(1 for h in hijos if h.id.split("#")[0] == p.id)
        print(f"  {p.endpoint:35} {contar_tokens(modelo, p.texto):5} tokens → {n} partes")

    ejemplo = next(h for h in hijos if h.id == "esco:post:insert-cuenta-comitente#3")
    print(f"\n--- {ejemplo.id} (primeras 8 líneas) ---")
    print("\n".join(ejemplo.texto.splitlines()[:8]))
