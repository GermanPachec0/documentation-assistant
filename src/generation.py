"""Generación: pregunta → búsqueda → contexto etiquetado → LLM → respuesta con fuentes.

Uso: PYTHONPATH=src uv run python src/generation.py
"""

import re
from collections.abc import Callable
from functools import cache

from rank_bm25 import BM25Okapi

from busqueda import _coleccion, _modelo, buscar_semantica, fusionar_rrf, tokenizar
from chunking import MAX_TOKENS_PADRE_AL_LLM, PRESUPUESTO_TOKENS_CONTEXTO, contar_tokens, encabezado_de
from ingest import PREFIJO_PREGUNTA, aplicar_glosario, cargar_padres
from llm import generar
from models import Fragmento

SISTEMA = """\
Sos un asistente que ayuda a desarrolladores a usar la API de ESCO (EscoAPI v9),
un sistema de back office bursátil argentino. No tiene nada que ver con la
clasificación europea de ocupaciones que también se llama ESCO.

Reglas:

1. Respondé SOLO con la información de las FUENTES que vienen en el mensaje.
   No uses lo que sepas de otras APIs ni de memoria. Si algo no está en las
   fuentes, no lo completes ni lo supongas.

2. Si NINGUNA fuente trata lo que se pregunta, empezá exactamente con:
   "No encontré ningún endpoint para eso en la documentación de ESCO."
   Esto vale también cuando la pregunta no tiene nada que ver con ESCO:
   aunque te lleguen fuentes, no las fuerces para que encajen.

3. Si el endpoint SÍ aparece pero la documentación no detalla lo que se
   pregunta (por ejemplo, sus parámetros), NO uses la frase de la regla 2.
   Decí: "El endpoint existe, pero la documentación no detalla <lo que se
   preguntó>." No inventes parámetros, tipos ni valores.

4. Citá cada dato con el número de su fuente, exactamente así: [FUENTE 2].
   Solo el número, entre corchetes; una cita por fuente ([FUENTE 1] [FUENTE 3]).
   No escribas vos el nombre del endpoint ni las páginas en la cita, y no
   agregues una lista de fuentes al final: eso lo agrega el sistema.

5. Elegí el endpoint por la ACCIÓN que pide el usuario. Varias fuentes pueden
   hablar del mismo tema con acciones opuestas (consultar, actualizar,
   eliminar; acreditar o debitar dinero). Usá el que corresponde a lo que se
   pidió, y en una línea explicá por qué no es el otro. Equivocarse de acción
   en un sistema financiero es grave: si no estás seguro, decilo.

6. Si la pregunta puede referirse a varias cosas distintas (por ejemplo,
   "operaciones" puede ser órdenes, boletos o movimientos), no elijas una al
   azar: nombrá las opciones con su endpoint y preguntá a cuál se refiere.

7. Cuando la respuesta sea un endpoint, terminá con un REQUEST DE EJEMPLO
   listo para copiar y pegar, con DATOS FICTICIOS:
   - CAMPOS: solo los que figuran en la fuente de ESE endpoint (request o
     tabla de parámetros). Nunca agregues un campo que no esté documentado.
     Incluí los obligatorios y, si son pocos, los opcionales más útiles.
   - VALORES: inventados pero verosímiles y del tipo correcto (decimal/long →
     número, string → texto, boolean → true/false, fecha →
     "2025-03-14T00:00:00"). Si la fuente lista los valores posibles de un
     campo (por ejemplo "PA" Particular), usá uno de esos.
   - SINTAXIS: JSON válido, con comillas dobles rectas, sin comas finales y
     sin comentarios adentro (un comentario rompe el JSON).
   - Si el PDF trae un ejemplo, usalo como base, pero con la sintaxis
     corregida si estaba rota, y decí qué corregiste. Reemplazá los valores
     de relleno ("string", 0) por valores ficticios verosímiles, y NUNCA
     copies emails, nombres, CUIT, CBU ni documentos del ejemplo del PDF:
     pueden ser datos reales. Inventá otros.
   - Si el request es muy grande, incluí los obligatorios y los bloques
     del caso elegido, y aclará que hay más campos opcionales en la fuente.
   - COHERENCIA: el ejemplo tiene que ser un caso posible. Si un campo
     define el tipo de caso (por ejemplo persona física o jurídica), incluí
     solo los bloques que corresponden a ese caso.
   - Si el endpoint es GET sin body, mostrá la ruta con los parámetros en la
     query string en un bloque ```http (NO ```json: una ruta no es JSON).
   Debajo del bloque, en una línea: "Datos ficticios. Armado a partir de la
   documentación, no probado contra la API real."
   No incluyas URL base ni autenticación: no están en las fuentes.
   Si la documentación no detalla los campos del endpoint (regla 3), NO
   armes request de ejemplo: un cuerpo vacío o adivinado no le sirve a nadie.

8. Si la pregunta pide un FLUJO o PROCESO completo (por ejemplo "dar de alta
   una cuenta con todo lo necesario"), armalo como pasos numerados, en orden:
   - Paso 1, consultas previas: si la fuente trae un bloque "[Consultas
     previas]", listá TODAS esas consultas (consulta → campos que completa),
     agrupadas por tema (cuenta, titulares, domicilio, cuenta bancaria,
     datos jurídicos…). Son las que dan los códigos válidos. Si una está
     marcada como no documentada, decilo.
   - Paso 2, el endpoint principal, con sus obligatorios y el request de
     ejemplo (regla 7).
   - Paso 3, si las fuentes lo muestran, los endpoints complementarios
     (por ejemplo, agregar una cuenta bancaria o un domicilio después).
   Usá solo lo que está en las fuentes: no agregues pasos que no figuren.

Formato:
- Si es un flujo, seguí la regla 8. Si no:
- Primero, la respuesta directa: qué endpoint y método usar, y por qué.
- Después, los parámetros obligatorios, con sus citas.
- Al final, el request de ejemplo, así:

  **Request de ejemplo** — `MÉTODO nombre-del-endpoint`
  ```json
  { ... }
  ```

Respondé en español, salvo que la pregunta esté en otro idioma.
"""

Buscador = Callable[[str, int], list[str]]


@cache
def _padres_por_id() -> dict[str, Fragmento]:
    # Los mismos padres que se ingestaron (con el glosario aplicado).
    return {p.id: p for p in aplicar_glosario(cargar_padres())}


@cache
def _hijos_de(padre_id: str) -> tuple[list[str], list[str], BM25Okapi]:
    """Ids y textos de los hijos de un padre, en orden (#1, #2…), y su índice BM25."""
    res = _coleccion().get(where={"$and": [{"padre_id": padre_id}, {"unidad": "hijo"}]}, include=["documents"])
    pares = sorted(zip(res["ids"], res["documents"]), key=lambda par: int(par[0].split("#")[1]))
    ids, textos = [i for i, _ in pares], [t for _, t in pares]
    return ids, textos, BM25Okapi([tokenizar(t) for t in textos])


def _mejor_hijo(padre_id: str, pregunta: str) -> int:
    """El número del hijo más relevante, fusionando semántica y BM25 con RRF.

    Solo con embeddings no alcanza: todos los hijos repiten el encabezado del
    padre y la similitud entre ellos difiere en 0.02 (casi azar). BM25 aporta
    la palabra exacta que los distingue ("jurídica" está en un solo hijo).
    """
    ids, _, bm25 = _hijos_de(padre_id)
    vector = _modelo().encode([PREFIJO_PREGUNTA + pregunta], normalize_embeddings=True)
    semantico = _coleccion().query(
        query_embeddings=vector.tolist(),
        n_results=len(ids),
        where={"$and": [{"padre_id": padre_id}, {"unidad": "hijo"}]},
        include=[],
    )["ids"][0]
    puntajes = bm25.get_scores(tokenizar(pregunta))
    por_palabras = [ids[i] for i in sorted(range(len(ids)), key=lambda i: puntajes[i], reverse=True)]
    mejor = fusionar_rrf([semantico, por_palabras], k=1)[0]
    return int(mejor.split("#")[1])


def _hijo_y_vecinos(padre: Fragmento, pregunta: str) -> Fragmento:
    """Para un padre gigante: el PRINCIPIO del endpoint + el hijo más relevante y sus vecinos.

    El hijo #1 va siempre: tiene la descripción y el comienzo del request,
    que el LLM necesita para casi cualquier pregunta sobre cómo llamarlo.
    Los hijos repiten el encabezado del padre: va una sola vez, al principio.
    """
    ids, textos, _ = _hijos_de(padre.id)
    total = len(ids)
    n = _mejor_hijo(padre.id, pregunta)
    elegidos = sorted({1} | {i for i in (n - 1, n, n + 1) if 1 <= i <= total})

    n_encabezado = len(encabezado_de(padre, _modelo()).texto.splitlines())
    lineas = textos[elegidos[0] - 1].splitlines()
    anterior = elegidos[0]
    for i in elegidos[1:]:
        if i != anterior + 1:
            lineas.append(f"[… se omiten las partes {anterior + 1}-{i - 1} …]" if i - anterior > 2
                          else f"[… se omite la parte {anterior + 1} …]")
        lineas += textos[i - 1].splitlines()[n_encabezado:]  # sin repetir el encabezado
        anterior = i

    # Las partes van en el id ("x#1,3-5de20") para que la etiqueta las muestre.
    return padre.model_copy(update={"texto": "\n".join(lineas), "id": f"{padre.id}#{_rangos(elegidos)}de{total}"})


def _rangos(numeros: list[int]) -> str:
    """[1, 3, 4, 5] → "1,3-5"."""
    grupos: list[list[int]] = []
    for x in numeros:
        if grupos and x == grupos[-1][-1] + 1:
            grupos[-1].append(x)
        else:
            grupos.append([x])
    return ",".join(str(g[0]) if len(g) == 1 else f"{g[0]}-{g[-1]}" for g in grupos)


# "Los valores posibles se pueden obtener con la consulta get-actividades":
# así marca el PDF que un campo se completa con el código de un catálogo.
_PROPIEDAD = re.compile(r"public\s+[\w<>?\[\],]+\s+(\w+)\s*\{")
_CONSULTA_GET = re.compile(r"\b(get-[A-Za-z0-9-]+)")


def _clave(endpoint: str) -> str:
    # get-actividadesUIF y get-actividades-uif son el mismo: sin guiones ni mayúsculas.
    return endpoint.lower().replace("-", "")


def consultas_previas(fragmento: Fragmento) -> list[tuple[str, str, bool]]:
    """(campo, consulta get-…, ¿está documentada en el PDF?), en el orden del texto.

    El código extrae, el LLM redacta: en un endpoint de 7.000 tokens el modelo
    no encuentra estas líneas sueltas (probado: decía "no hay consultas
    previas"). Sacadas por código y puestas al principio, no se le escapan.
    """
    documentados = {_clave(p.endpoint): p.endpoint for p in _padres_por_id().values()}
    campo, vistos, salida = None, set(), []
    for linea in fragmento.texto.splitlines():
        if m := _PROPIEDAD.search(linea):
            campo = m.group(1)
            continue
        for consulta in _CONSULTA_GET.findall(linea):
            consulta = consulta.rstrip("-.")
            clave = _clave(consulta)
            if not campo or clave == _clave(fragmento.endpoint) or (campo, clave) in vistos:
                continue
            vistos.add((campo, clave))
            salida.append((campo, documentados.get(clave, consulta), clave in documentados))
    return salida


def _con_consultas_previas(fragmento: Fragmento, padre: Fragmento) -> Fragmento:
    """Agrega, después del encabezado, la lista de campos que salen de un catálogo."""
    consultas = consultas_previas(padre)
    if not consultas:
        return fragmento
    lineas = ["[Consultas previas: campos cuyo valor sale de otra consulta, en el orden del documento]"]
    for campo, consulta, documentada in consultas:
        lineas.append(f"- {campo} → {consulta}" + ("" if documentada else " (nombrada, pero no documentada en este PDF)"))
    lineas.append("[Fin consultas previas]")
    texto = fragmento.texto.splitlines()
    n = len(encabezado_de(fragmento, _modelo()).texto.splitlines())
    return fragmento.model_copy(update={"texto": "\n".join(texto[:n] + lineas + texto[n:])})


def armar_contexto(pregunta: str, k: int = 5, buscar: Buscador = buscar_semantica) -> list[Fragmento]:
    """Los k padres que trae la búsqueda, en orden (el más relevante primero:
    los LLMs atienden más al principio del contexto que al medio).

    Decisión 3 (chunking.py): el padre completo si es chico o si todavía entra
    en el presupuesto del contexto; si no, el principio + el hijo más relevante
    y sus vecinos. Las consultas previas se sacan del padre COMPLETO, aunque al
    LLM le llegue un recorte.
    """
    padres = _padres_por_id()
    fragmentos = []
    restante = PRESUPUESTO_TOKENS_CONTEXTO
    for padre_id in buscar(pregunta, k):
        padre = padres[padre_id]
        tokens = contar_tokens(_modelo(), padre.texto)
        if tokens <= MAX_TOKENS_PADRE_AL_LLM or tokens <= restante:
            fragmento = padre
        else:
            fragmento = _hijo_y_vecinos(padre, pregunta)
            tokens = contar_tokens(_modelo(), fragmento.texto)
        restante -= tokens
        fragmentos.append(_limpiar_ejemplos(_con_consultas_previas(fragmento, padre)))
    return fragmentos


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def _limpiar_ejemplos(fragmento: Fragmento) -> Fragmento:
    """Los ejemplos del PDF traen emails que parecen de personas reales
    y valores de relleno ("string").

    El prompt le pide al LLM que no los copie, pero Flash-Lite a veces lo hace
    igual. Lo que no tiene que salir nunca se resuelve por código: el email
    no llega al modelo, así que no lo puede repetir.
    """
    texto = _EMAIL.sub("usuario@ejemplo.com", fragmento.texto)
    # "string" es el relleno por defecto de Swagger: el modelo lo copiaba
    # tal cual. Marcado así, lo reconoce como un hueco a completar.
    texto = texto.replace('"string"', '"<INVENTAR VALOR>"')
    return fragmento if texto == fragmento.texto else fragmento.model_copy(update={"texto": texto})


def _etiqueta(f: Fragmento) -> str:
    if f.tipo == "nota":
        que = f"NOTA {f.endpoint}"
    else:
        que = f"{f.metodo} {f.endpoint}"
    if "#" in f.id:  # recorte de un padre gigante: "x#3-5de20"
        rango, total = f.id.split("#")[1].split("de")
        que += f" (partes {rango} de {total}: recorte de un endpoint largo)"
    if f.pagina_inicio:
        paginas = (
            f"pág. {f.pagina_inicio}"
            if f.pagina_inicio == f.pagina_fin
            else f"págs. {f.pagina_inicio}-{f.pagina_fin}"
        )
        donde = f"{paginas} del PDF ({f.fuente})"
    else:
        donde = f.fuente
    return f"{f.sistema} | {que} | {donde}"


def formatear_contexto(fragmentos: list[Fragmento]) -> str:
    bloques = [
        f"[FUENTE {n}] {_etiqueta(f)}\n{f.texto}\n[FIN FUENTE {n}]"
        for n, f in enumerate(fragmentos, start=1)
    ]
    return "\n\n".join(bloques)


_CITA = re.compile(r"\[FUENTE\s+(\d+)\]")


def formatear_citas(texto: str, fragmentos: list[Fragmento]) -> tuple[str, list[int]]:
    """El LLM elige, el código formatea.

    El modelo escribe [FUENTE n]; acá se reemplaza por la etiqueta real y se
    arma la lista de fuentes citadas. Así una cita nunca mezcla el endpoint
    de una fuente con las páginas de otra. Un número que no existe queda
    marcado, en vez de convertirse en una cita falsa.
    """
    citadas: list[int] = []

    def reemplazar(m: re.Match) -> str:
        n = int(m.group(1))
        if not 1 <= n <= len(fragmentos):
            return f"[FUENTE {n} ⚠ no existe]"
        if n not in citadas:
            citadas.append(n)
        return f"[{n}]"

    cuerpo = _CITA.sub(reemplazar, texto).rstrip()
    return cuerpo, citadas


def lista_fuentes(fragmentos: list[Fragmento], citadas: list[int]) -> str:
    """La lista "Fuentes:" que va al pie de la respuesta (solo las citadas)."""
    if not citadas:
        return ""
    return "Fuentes:\n" + "\n".join(f"[{n}] {_etiqueta(fragmentos[n - 1])}" for n in sorted(citadas))


_PALABRAS_INGLES = {"how", "what", "which", "do", "does", "is", "are", "the", "can", "i", "to", "of", "a", "an", "where", "when", "get", "create", "update", "delete"}


def _idioma(pregunta: str) -> str:
    """Inglés o español, por palabras frecuentes. Simple a propósito.

    La regla "respondé en el idioma de la pregunta" del sistema no alcanzó:
    con miles de tokens de fuentes en español, el modelo respondía en español.
    Decírselo explícito, al final, sí funciona.
    """
    palabras = re.findall(r"[a-záéíóúñ]+", pregunta.lower())
    en_ingles = sum(p in _PALABRAS_INGLES for p in palabras)
    return "inglés" if palabras and en_ingles / len(palabras) >= 0.3 else "español"


def responder(pregunta: str, k: int = 5, buscar: Buscador = buscar_semantica) -> dict:
    fragmentos = armar_contexto(pregunta, k, buscar)
    prompt = (
        "FUENTES:\n\n"
        + formatear_contexto(fragmentos)
        + f"\n\nPregunta del usuario: {pregunta}"
        # Al final, cerca de la pregunta: con 10.000+ tokens de fuentes en el
        # medio, el modelo se olvidaba de la regla de idioma del sistema.
        + f"\n\n(Respondé en {_idioma(pregunta)}.)"
    )
    r = generar(prompt, sistema=SISTEMA)
    cuerpo, citadas = formatear_citas(r["texto"], fragmentos)
    pie = lista_fuentes(fragmentos, citadas)
    return {
        "pregunta": pregunta,
        "respuesta": cuerpo + ("\n\n" + pie if pie else ""),  # texto completo (consola)
        "cuerpo": cuerpo,  # sin la lista de fuentes (la interfaz la muestra aparte)
        # Todo lo que se le mandó al LLM, marcando lo que citó: para la interfaz
        # y para depurar ("¿qué vio el modelo cuando respondió esto?").
        "contexto": [
            {"n": n, "id": f.id, "etiqueta": _etiqueta(f), "texto": f.texto, "citada": n in citadas}
            for n, f in enumerate(fragmentos, start=1)
        ],
        "respuesta_cruda": r["texto"],
        "fuentes": [f.id for f in fragmentos],
        "fuentes_citadas": [fragmentos[n - 1].id for n in citadas],
        "tokens_entrada": r["tokens_entrada"],
        "tokens_salida": r["tokens_salida"],
        "segundos": r["segundos"],
        "modelo": r["modelo"],
    }


if __name__ == "__main__":
    PRUEBAS = [
        "¿Cómo actualizo el domicilio de un comitente?",  # tiene que elegir el PUT
        "¿Qué parámetros recibe update-condominio?",  # el PDF no los documenta
        "Dame la receta del pan casero",  # fuera de tema
        "¿Qué FCI puedo usar para operar?",  # sale de la nota
    ]
    for pregunta in PRUEBAS:
        r = responder(pregunta)
        print("=" * 80)
        print(f"PREGUNTA: {r['pregunta']}")
        print(f"FUENTES:  {', '.join(r['fuentes'])}")
        print(f"TOKENS:   {r['tokens_entrada']} entrada / {r['tokens_salida']} salida   ({r['segundos']} s, {r['modelo']})")
        print("-" * 80)
        print(r["respuesta"])
