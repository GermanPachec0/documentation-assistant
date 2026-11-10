import json
import re
import unicodedata
from pathlib import Path

import pymupdf

from models import Fragmento

# "HTTP request: <MÉTODO> <separador> <nombre>"
# - el separador puede ser "->" o "-" (ej. "PUT - Riesgo-cuenta-comitente")
# - el nombre es el resto de la línea: puede traer espacios y guiones largos
#   ("insert – oferta-orden", "get-all cotizaciones-monedas."), así que no
#   alcanza con \S+; lo limpio después.
_TITULO = re.compile(r"HTTP request:\s*(\w+)\s*-+>?\s*(.+)")


def parsear_titulo(texto: str) -> tuple[str, str]:
    """Devuelve (método, endpoint) normalizados a partir de un título del PDF.

    Tira ValueError si el texto no tiene forma de título: prefiero que el
    lector explote y me muestre el título raro a que lo saltee en silencio.
    """
    match = _TITULO.search(texto)
    if match is None:
        raise ValueError(f"No parece un título de endpoint: {texto!r}")

    metodo = match.group(1).upper()

    nombre = match.group(2).strip().rstrip(".")
    nombre = nombre.replace("–", "-")  # guion largo → guion normal
    nombre = re.sub(r"\s*-\s*", "-", nombre)  # "insert - orden" → "insert-orden"
    nombre = re.sub(r"\s+", "-", nombre)  # espacios que quedan → guion
    endpoint = nombre.lower()

    return metodo, endpoint


# Medido sobre las 270 páginas (todas de 792 pt de alto):
#   encabezado "primary.com … EscoAPI v9"  y0 =  36.2  (0.046)  color 0x1c1482
#   número de página                       y0 = 710.4  (0.897)  Calibri negro
#   pie "Tecnología expansiva"             y0 = 722.0  (0.912)  color 0x1c1482
#   contenido real más bajo                y0 = 696.7  (0.880)  pág. idx 139
#   línea de solo dígitos más baja que NO es número de página: y0/h = 0.873
# El umbral va en el hueco entre 0.880 y 0.897.
COLOR_ENCABEZADO_PIE = 0x1C1482
UMBRAL_PIE = 0.885


def es_ruido(linea: dict, alto_pagina: float) -> bool:
    """True si la línea (de page.get_text("dict")) es encabezado, pie o n° de página."""
    if any(span["color"] == COLOR_ENCABEZADO_PIE for span in linea["spans"]):
        return True

    texto = "".join(span["text"] for span in linea["spans"]).strip()
    y0 = linea["bbox"][1]
    # Las dos condiciones juntas: solo posición borraría contenido que caiga
    # abajo; solo dígitos borraría un "0" de un JSON o los números del índice.
    return texto.isdigit() and y0 / alto_pagina > UMBRAL_PIE


# Medido sobre las 198 apariciones de "HTTP request:":
#   146 en 0x0070c0, 20 en 0x2f5496, 1 con "●" negro + texto en 0x0000ff
#   → títulos reales (167): el span que dice "HTTP request" NO es negro
#    31 en negro (0x000000) → son el índice impreso, no títulos
NEGRO = 0x000000

# Dos niveles de encabezado, medidos en las págs. 195, 228 y 239:
#
# Sección = el grupo. Va al campo `seccion`.
#   0x1f3863 Calibri-Bold 12      → "Administrar Cuenta Comitente"
ESTILOS_SECCION = {(0x1F3863, "Calibri-Bold")}
#
# Subtítulo = el nombre "humano" de un endpoint. Va DENTRO del texto, porque
# se parece a la pregunta del usuario ("Cancelar Orden").
#   0x2f5496 Calibri-Italic 11    → "Consulta de Actividades"     (pág. 195)
#   0x2f5496 Calibri 11 (regular) → "Cancelar Orden"              (pág. 239)
# Ojo: 0x2f5496 Calibri regular es TAMBIÉN el estilo de 20 títulos
# "HTTP request:". Por eso es_titulo se evalúa siempre antes que es_subtitulo.
ESTILOS_SUBTITULO = {(0x2F5496, "Calibri-Italic"), (0x2F5496, "Calibri")}

# Antes de esta página (visor) están la portada, el índice, el changelog y
# la autenticación: no hay endpoints. Ahí no se arman preámbulos.
PRIMERA_PAGINA_CUERPO = 24

SISTEMA = "esco"

# El modelo de embeddings no sabe que PUT significa "actualizar". Se agregan
# verbos en la primera línea: "PUT domicilio-cuenta-comitente (actualizar, modificar)".
#
# OJO: en este PDF el MÉTODO no dice la acción: 37 consultas usan POST
# (POST get-boletos) y 2 altas usan GET (GET insert-cotizacion-cierre-monedas).
# La señal confiable es el PREFIJO del nombre; el método se usa solo cuando
# el nombre no tiene prefijo de acción (domicilio-…, persona, riesgo-…).
# Listas cortas y genéricas a propósito: nada específico de una pregunta
# del YAML (ej. "saldo"), para no optimizar para el examen.
VERBOS_POR_PREFIJO = {
    "get": "consultar, obtener",
    "esta": "consultar, obtener",
    "searchaccount": "consultar, buscar",
    "insert": "crear, dar de alta, registrar",
    "update": "actualizar, modificar",
    "delete": "eliminar, borrar",
    "cancel": "cancelar, anular",
    "cancelar": "cancelar, anular",
    "previsualizar": "previsualizar, simular",
    "cambiar": "cambiar, modificar",
}
VERBOS_POR_METODO = {
    "GET": "consultar, obtener",
    "POST": "enviar",
    "PUT": "actualizar, modificar",
    "PATCH": "actualizar, modificar",
    "DELETE": "eliminar, borrar",
}


def verbos_de(metodo: str, endpoint: str) -> str:
    prefijo = endpoint.split("-")[0]
    return VERBOS_POR_PREFIJO.get(prefijo) or VERBOS_POR_METODO[metodo]

# Endpoints documentados dos veces en el PDF. Decidido a mano, comparando textos:
DESCARTAR = {
    # get-actividades: me quedo con la pág. 195. La 187 no tiene descripción
    # ("Esta consulta devuelve la lista de actividades habilitadas.") ni
    # "No utiliza request.", y sus campos dicen solo "Código interno." /
    # "Descripción." en vez de "... de la Actividad".
    ("esco:get:get-actividades", 187),
    # get-tipos-contrib-retencion: me quedo con la pág. 203. La 215 declara
    # `long Codigo`, pero los valores del ejemplo son "EX", "EJ", "IN", "NI":
    # el tipo correcto es `string`, que es lo que dice la 203.
    # (La 215 tiene una descripción más completa; se pierde.)
    ("esco:get:get-tipos-contrib-retencion", 215),
}


def _texto_linea(linea: dict) -> str:
    return "".join(span["text"] for span in linea["spans"]).strip()


def es_titulo(linea: dict) -> bool:
    # "● HTTP request: DELETE - domicilio-cuenta-comitente" (pág. 40) NO es un
    # título: es un ítem de una lista con viñetas que enumera los endpoints de
    # la sección. El DELETE real está en la pág. 42.
    if _texto_linea(linea).startswith("●"):
        return False
    return any(
        "HTTP request:" in span["text"] and span["color"] != NEGRO
        for span in linea["spans"]
    )


def _estilo(linea: dict) -> tuple[int, str] | None:
    spans = [s for s in linea["spans"] if s["text"].strip()]
    return (spans[0]["color"], spans[0]["font"]) if spans else None


def es_seccion(linea: dict) -> bool:
    return _estilo(linea) in ESTILOS_SECCION


def es_subtitulo(linea: dict) -> bool:
    return _estilo(linea) in ESTILOS_SUBTITULO


def leer_pdf(
    ruta: str | Path,
    descartadas: list[tuple[int, str]] | None = None,
    huerfanos: list[tuple[int, list[str]]] | None = None,
) -> list[Fragmento]:
    """Lee el PDF y devuelve un Fragmento por endpoint.

    Estructura que se espera en el PDF:
        [Sección]                    → campo `seccion`
        [Subtítulo]                  ┐
        [descripción…]               ┘ preámbulo: va al endpoint SIGUIENTE
        HTTP request: MÉTODO -> x    → abre el fragmento
        [cuerpo…]

    Diagnóstico opcional:
    - `descartadas`: (página, texto) de las líneas que no quedaron en ningún fragmento.
    - `huerfanos`: (página, líneas) de preámbulos que nunca llegaron a un endpoint.
    """
    ruta = Path(ruta)
    pdf = pymupdf.open(ruta)

    fragmentos: list[Fragmento] = []
    vistos: set[str] = set()
    seccion = ""
    subtitulo_vigente: str | None = None  # último subtítulo visto en esta sección
    preambulo: dict | None = None  # subtítulo + descripción a la espera de un título
    actual: dict | None = None  # endpoint abierto: metodo, endpoint, paginas, lineas

    def cerrar_preambulo() -> None:
        # Un preámbulo que no llegó a ningún HTTP request:
        # - con contenido (subtítulo + texto) → es una nota de negocio, ej.
        #   "Validación de FCI para operar" (pág. 224) u "Órdenes a precio de
        #   mercado" (pág. 245). Se guarda como fragmento propio.
        # - solo el subtítulo → es un título de grupo ("Administración de
        #   Órdenes"), seguido por otro subtítulo. Se registra, no se tira callado.
        nonlocal preambulo
        if preambulo is None:
            return
        lineas = preambulo["lineas"]
        if len(lineas) > 1:
            slug = _slug(lineas[0])
            fragmentos.append(
                Fragmento(
                    id=f"{SISTEMA}:nota:{slug}",
                    tipo="nota",
                    sistema=SISTEMA,
                    endpoint=slug,
                    seccion=preambulo["seccion"],
                    texto="\n".join(["NOTA " + lineas[0], *lineas[1:]]),
                    fuente=ruta.name,
                    pagina_inicio=preambulo["pagina"],
                    pagina_fin=preambulo["pagina_fin"],
                )
            )
        elif huerfanos is not None:
            huerfanos.append((preambulo["pagina"], lineas))
        preambulo = None

    def cerrar() -> None:
        nonlocal actual
        if actual is None:
            return
        # El método va en el id: domicilio-cuenta-comitente existe como
        # DELETE, GET y PUT, y son endpoints distintos.
        id_ = f"{SISTEMA}:{actual['metodo'].lower()}:{actual['endpoint']}"
        if (id_, actual["pagina_inicio"]) in DESCARTAR:
            actual = None
            return
        if id_ in vistos:
            # Un duplicado que no está en DESCARTAR: hay que compararlo a mano.
            raise ValueError(
                f"Endpoint duplicado sin decidir: {id_} (pág. {actual['pagina_inicio']}). "
                "Compará las versiones y agregá la que sobra a DESCARTAR."
            )
        vistos.add(id_)

        verbos = verbos_de(actual["metodo"], actual["endpoint"])
        partes = [f"{actual['metodo']} {actual['endpoint']} ({verbos})"]
        if actual["preambulo"]:
            partes += actual["preambulo"]  # subtítulo propio + descripción
        elif actual["subtitulo"]:
            # Sin subtítulo propio: hereda el del bloque en el que está
            # (ej. get-condominios → "Modificación de Cuentas Comitente").
            partes.append(actual["subtitulo"])
        partes += actual["lineas"]

        fragmentos.append(
            Fragmento(
                id=id_,
                sistema=SISTEMA,
                endpoint=actual["endpoint"],
                metodo=actual["metodo"],
                seccion=actual["seccion"],
                texto="\n".join(partes),
                fuente=ruta.name,
                pagina_inicio=actual["pagina_inicio"],
                pagina_fin=actual["pagina_fin"],
            )
        )
        actual = None

    for indice, page in enumerate(pdf):
        pagina = indice + 1  # numeración del visor
        alto = page.rect.height
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for linea in block["lines"]:
                texto = _texto_linea(linea)
                if not texto or es_ruido(linea, alto):
                    continue
                en_cuerpo = pagina >= PRIMERA_PAGINA_CUERPO

                # El orden de los chequeos importa: es_titulo va PRIMERO porque
                # 20 títulos "HTTP request:" tienen el mismo estilo que un
                # subtítulo (0x2f5496 Calibri). Si no, serían preámbulos.
                if es_titulo(linea):
                    cerrar()
                    metodo, endpoint = parsear_titulo(texto)
                    actual = {
                        "metodo": metodo,
                        "endpoint": endpoint,
                        "seccion": seccion,
                        "subtitulo": subtitulo_vigente,
                        "preambulo": preambulo["lineas"] if preambulo else [],
                        "pagina_inicio": preambulo["pagina"] if preambulo else pagina,
                        "pagina_fin": pagina,
                        "lineas": [],
                    }
                    preambulo = None
                elif es_seccion(linea):
                    cerrar()
                    cerrar_preambulo()
                    seccion = texto
                    subtitulo_vigente = None
                elif es_subtitulo(linea) and en_cuerpo:
                    # Un subtítulo cierra el endpoint anterior y abre un preámbulo
                    # para el SIGUIENTE: lo que viene ahora es su descripción.
                    cerrar()
                    cerrar_preambulo()
                    preambulo = {
                        "pagina": pagina,
                        "pagina_fin": pagina,
                        "seccion": seccion,
                        "lineas": [texto],
                    }
                    subtitulo_vigente = texto
                elif actual is not None:
                    actual["lineas"].append(texto)
                    actual["pagina_fin"] = pagina
                elif preambulo is not None:
                    preambulo["lineas"].append(texto)
                    preambulo["pagina_fin"] = pagina
                elif descartadas is not None:
                    descartadas.append((pagina, texto))

    cerrar()  # el último endpoint del documento
    cerrar_preambulo()  # un preámbulo al final sin endpoint también se resuelve
    return fragmentos


def _slug(texto: str) -> str:
    """'Validación de FCI para operar' → 'validacion-de-fci-para-operar'."""
    sin_tildes = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", sin_tildes.lower()).strip("-")


def guardar_json(fragmentos: list[Fragmento], ruta: str | Path) -> None:
    """Guarda los fragmentos como lista JSON. Es lo que lee la ingesta (Paso 4)."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    datos = [f.model_dump() for f in fragmentos]
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    casos = {
        "HTTP request: POST  ->GET-CONDOMINIOS ": ("POST", "get-condominios"),
        "HTTP request: GET ->GET-Persona": ("GET", "get-persona"),
        "HTTP request: GET  -> get-firmantes": ("GET", "get-firmantes"),
        # Casos raros sacados de recorrer las 270 páginas:
        "HTTP request: PUT - Riesgo-cuenta-comitente ": ("PUT", "riesgo-cuenta-comitente"),
        "HTTP request: POST  -> insert – oferta-orden ": ("POST", "insert-oferta-orden"),
        "HTTP request: POST  -> cancelar– garantias-orden ": ("POST", "cancelar-garantias-orden"),
        "HTTP request: GET  -> get-cotizaciones-monedas. ": ("GET", "get-cotizaciones-monedas"),
        "HTTP request: GET  -> get-all cotizaciones-monedas. ": ("GET", "get-all-cotizaciones-monedas"),
        "HTTP request: GET -> get-resultados-no- realizados ": ("GET", "get-resultados-no-realizados"),
        "● HTTP request: DELETE - domicilio-cuenta-comitente ": ("DELETE", "domicilio-cuenta-comitente"),
    }
    for entrada, esperado in casos.items():
        obtenido = parsear_titulo(entrada)
        print("OK " if obtenido == esperado else "MAL", repr(entrada), "→", obtenido)

    try:
        parsear_titulo("Request:")
    except ValueError as e:
        print("OK  error esperado →", e)

    # es_ruido sobre todo el PDF: cuántas líneas descarta y de qué tipo
    import collections

    import pymupdf

    pdf = pymupdf.open("data/raw/Instructivo_Esco_Api_escobolsa_v9.pdf")
    descartadas = collections.Counter()
    for page in pdf:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                if es_ruido(line, page.rect.height):
                    texto = "".join(s["text"] for s in line["spans"]).strip()
                    clave = "<número de página>" if texto.isdigit() else texto[:30]
                    descartadas[clave] += 1
    print("\nes_ruido sobre", pdf.page_count, "páginas:")
    for texto, n in descartadas.most_common():
        print(f"  {n:4} × {texto!r}")

    # --- leer_pdf ---
    fragmentos = leer_pdf("data/raw/Instructivo_Esco_Api_escobolsa_v9.pdf")

    # 1. Cantidad vs. apariciones de "HTTP request:"
    apariciones = sum(
        page.get_text().count("HTTP request:") for page in pdf
    )
    endpoints = [f for f in fragmentos if f.tipo == "endpoint"]
    notas = [f for f in fragmentos if f.tipo == "nota"]
    print(
        f"\nFragmentos: {len(fragmentos)} = {len(endpoints)} endpoints + {len(notas)} notas"
        f"  (apariciones de 'HTTP request:': {apariciones})"
    )
    for n in notas:
        print(f"  {n.id}  págs. {n.pagina_inicio}-{n.pagina_fin}  ({len(n.texto)} chars)")

    # 2. ids repetidos
    repetidos = {
        id_: n for id_, n in collections.Counter(f.id for f in fragmentos).items() if n > 1
    }
    print(f"ids repetidos: {len(repetidos)}")
    for id_, n in sorted(repetidos.items()):
        metodos = [f.metodo for f in fragmentos if f.id == id_]
        print(f"  {n} × {id_}  {metodos}")

    # 4. Líneas descartadas (fuera de todo fragmento)
    descartadas: list[tuple[int, str]] = []
    huerfanos: list[tuple[int, list[str]]] = []
    leer_pdf("data/raw/Instructivo_Esco_Api_escobolsa_v9.pdf", descartadas, huerfanos)

    # 6. Preámbulos huérfanos (subtítulo sin HTTP request: después)
    print(f"\nPreámbulos huérfanos: {len(huerfanos)}")
    for p, lineas in huerfanos:
        print(f"  pág. {p:3} | {' ⏎ '.join(lineas)[:110]}")
    # Antes de la pág. 24 está la portada, el índice y el changelog: ahí es
    # esperable descartar todo. Lo que importa es lo que cae después.
    en_cuerpo = [(p, t) for p, t in descartadas if p >= 24]
    print(f"\nLíneas descartadas: {len(descartadas)} en total, {len(en_cuerpo)} desde la pág. 24")
    for p, t in en_cuerpo[:10]:
        print(f"  pág. {p:3} | {t[:80]}")

    # 5. Colas sospechosas: últimas 2 líneas de cada fragmento + quién sigue
    print("\nColas de fragmento (últimas 2 líneas → endpoint siguiente):")
    for actual_, siguiente in zip(fragmentos, fragmentos[1:]):
        cola = " ⏎ ".join(l[:45] for l in actual_.texto.splitlines()[-2:])
        print(f"  {actual_.endpoint[:30]:30} | {cola:95} → {siguiente.endpoint}")

    # 3. get-condominios contra el test dorado
    condominios = next(f for f in fragmentos if f.endpoint == "get-condominios")
    print("\n" + condominios.model_dump_json(indent=2, exclude={"texto"}))
    print("--- texto ---")
    print(condominios.texto)
    assert (condominios.metodo, condominios.pagina_inicio, condominios.pagina_fin) == (
        "POST",
        49,
        50,
    )
    assert "primary.com" not in condominios.texto
    assert not any(l.strip().isdigit() for l in condominios.texto.splitlines())
    assert "Modificación de Cuentas Comitente" in condominios.texto
    print("--- OK: POST, páginas 49–50, sin encabezados ni números sueltos, con subtítulo ---")

    # 7. Verificaciones del preámbulo
    por_endpoint = {f.endpoint: f for f in endpoints}
    assert len(endpoints) == 164, len(endpoints)
    assert [n.endpoint for n in notas] == [
        "validacion-de-fci-para-operar",
        "ordenes-a-precio-de-mercado",
    ], [n.endpoint for n in notas]
    assert "Insert de órdenes de venta" not in por_endpoint["insert-orden-compra"].texto
    for e in ("insert-orden-venta", "cancel-orden", "get-actividades"):
        print(f"\n--- comienzo de {e} ---")
        print("\n".join(por_endpoint[e].texto.splitlines()[:4]))
    assert por_endpoint["get-actividades"].texto.splitlines()[1:3] == [
        "Consulta de Actividades",
        "Esta consulta devuelve la lista de actividades habilitadas.",
    ]
    print("\n--- OK: 164 endpoints + 2 notas, preámbulos en su lugar ---")

    # 8. Guardar para la ingesta (Paso 4)
    salida = Path("data/chunks/esco.json")
    guardar_json(fragmentos, salida)
    releidos = [Fragmento(**d) for d in json.loads(salida.read_text(encoding="utf-8"))]
    assert releidos == fragmentos
    print(f"\nGuardados {len(fragmentos)} fragmentos en {salida} ({salida.stat().st_size:,} bytes)")
