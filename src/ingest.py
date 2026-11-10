"""Ingesta: fragmentos de data/chunks/*.json → embeddings → Chroma.

Uso: PYTHONPATH=src uv run python src/ingest.py
"""

import json
from pathlib import Path

import chromadb
import yaml
from sentence_transformers import SentenceTransformer

from chunking import SUFIJO_ENCABEZADO, encabezado_de, partir_largos
from models import Fragmento

DIR_CHUNKS = Path("data/chunks")
DIR_CHROMA = Path("data/chroma")
RUTA_GLOSARIO = Path("data/glosario.yaml")
COLECCION = "docs"
MODELO = "intfloat/multilingual-e5-small"
PREFIJO_DOC = "passage: "
PREFIJO_PREGUNTA = "query: "

# Chroma no acepta None en metadatos. Valores "no aplica":
SIN_METODO = ""  # las notas no tienen método (tipo="nota" ya lo dice)
SIN_PAGINA = 0  # las páginas arrancan en 1, así que 0 = "esta fuente no tiene páginas" (OpenAPI)


def cargar_padres(directorio: Path = DIR_CHUNKS) -> list[Fragmento]:
    """Todos los fragmentos de todas las fuentes (esco.json, y las que vengan)."""
    padres: list[Fragmento] = []
    for archivo in sorted(directorio.glob("*.json")):
        datos = json.loads(archivo.read_text(encoding="utf-8"))
        padres += [Fragmento(**d) for d in datos]
        print(f"  {archivo.name}: {len(datos)} fragmentos")
    return padres


def aplicar_glosario(padres: list[Fragmento], ruta: Path = RUTA_GLOSARIO) -> list[Fragmento]:
    """Agrega 'También: <términos>' como 2.ª línea del texto de cada padre del glosario.

    Va en la 2.ª línea para quedar dentro del encabezado (y por lo tanto en
    el vector #encabezado y en todos los hijos). No toca data/chunks/*.json:
    el glosario es conocimiento de negocio, no contenido del PDF.
    """
    if not ruta.exists():
        return padres
    glosario: dict[str, list[str]] = yaml.safe_load(ruta.read_text(encoding="utf-8")) or {}
    ids = {p.id for p in padres}
    desconocidos = sorted(set(glosario) - ids)
    if desconocidos:
        raise ValueError(f"{ruta}: ids que no existen en data/chunks: {desconocidos}")

    resultado = []
    for p in padres:
        if p.id in glosario:
            primera, *resto = p.texto.splitlines()
            linea = "También: " + ", ".join(glosario[p.id])
            p = p.model_copy(update={"texto": "\n".join([primera, linea, *resto])})
        resultado.append(p)
    print(f"  glosario: {len(glosario)} endpoints con términos de negocio")
    return resultado


def a_metadatos(hijo: Fragmento) -> dict[str, str | int]:
    return {
        "tipo": hijo.tipo,
        "sistema": hijo.sistema,
        "endpoint": hijo.endpoint,
        "metodo": hijo.metodo or SIN_METODO,
        "seccion": hijo.seccion,
        "fuente": hijo.fuente,
        "pagina_inicio": hijo.pagina_inicio or SIN_PAGINA,
        "pagina_fin": hijo.pagina_fin or SIN_PAGINA,
        # Para la decisión 3 (chunking.py): desde un hijo se vuelve al padre.
        "padre_id": hijo.id.split("#")[0],
        # "encabezado" (vector con solo método + subtítulo + descripción) o
        # "hijo" (una parte del texto completo). BM25 indexa solo los hijos.
        "unidad": "encabezado" if hijo.id.endswith(SUFIJO_ENCABEZADO) else "hijo",
    }


def abrir_coleccion() -> chromadb.Collection:
    cliente = chromadb.PersistentClient(path=str(DIR_CHROMA))
    coleccion = cliente.get_or_create_collection(
        COLECCION,
        # Distancia coseno: Chroma devuelve 1 − similitud.
        configuration={"hnsw": {"space": "cosine"}},
        # Los embeddings los calculamos nosotros: que Chroma no traiga su modelo.
        embedding_function=None,
    )
    # Trampa de get_or_create: si la colección ya existía con otra
    # configuración, la abre con la VIEJA e ignora la nueva. Se verifica.
    espacio = coleccion.configuration["hnsw"]["space"]
    if espacio != "cosine":
        raise RuntimeError(
            f"La colección '{COLECCION}' usa espacio '{espacio}', no 'cosine'. "
            f"Borrá {DIR_CHROMA} y volvé a correr la ingesta."
        )
    return coleccion


def ingestar() -> tuple[chromadb.Collection, list[Fragmento]]:
    print("Cargando fragmentos:")
    padres = aplicar_glosario(cargar_padres())
    modelo = SentenceTransformer(MODELO)
    hijos = [h for p in padres for h in partir_largos(p, modelo)]
    # Además de los hijos, un vector "limpio" por padre con solo su encabezado.
    encabezados = [encabezado_de(p, modelo) for p in padres]
    print(f"{len(padres)} padres → {len(hijos)} hijos + {len(encabezados)} encabezados")
    hijos = hijos + encabezados

    # normalize_embeddings=True: con vectores de largo 1, coseno = producto
    # punto. Chroma con espacio coseno ya normaliza por dentro, así que hoy
    # no cambia el resultado; pero así los vectores guardados ya son
    # comparables con @ (como en scratch/similitud.py) y no dependemos de que
    # quien los use después se acuerde de normalizar (ej. si se cambia a "ip").
    embeddings = modelo.encode(
        [PREFIJO_DOC + h.texto for h in hijos],
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    )

    coleccion = abrir_coleccion()
    ids = [h.id for h in hijos]

    # upsert y no add: si la ingesta se corre dos veces, add fallaría (o
    # duplicaría, según la versión) con los ids que ya existen; upsert los
    # reemplaza. Correr la ingesta N veces deja el mismo resultado.
    coleccion.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=[h.texto for h in hijos],
        metadatas=[a_metadatos(h) for h in hijos],
    )

    # upsert no borra lo que ya no existe: si un padre antes tenía 20 partes y
    # ahora 19, la #20 vieja quedaría colgada. Se borra lo que sobra.
    sobrantes = sorted(set(coleccion.get(include=[])["ids"]) - set(ids))
    if sobrantes:
        coleccion.delete(ids=sobrantes)
    print(f"Borrados {len(sobrantes)} ids que ya no existen")

    return coleccion, hijos


if __name__ == "__main__":
    coleccion, hijos = ingestar()

    total = coleccion.count()
    print(f"\ncoleccion.count() = {total}   espacio = {coleccion.configuration['hnsw']['space']}")
    # No "== 317": eso se rompe el día que se agregue otra fuente.
    assert total == len(hijos), (total, len(hijos))

    # Prueba de humo: la pregunta lleva "query: ", no "passage: "
    pregunta = "¿cómo actualizo el domicilio de un comitente?"
    modelo = SentenceTransformer(MODELO)
    vector = modelo.encode([PREFIJO_PREGUNTA + pregunta], normalize_embeddings=True)
    res = coleccion.query(query_embeddings=vector.tolist(), n_results=5)

    print(f"\nPregunta: {pregunta}")
    print(f"  {'distancia':>9} {'1-dist':>7}  id")
    for id_, dist in zip(res["ids"][0], res["distances"][0]):
        print(f"  {dist:9.3f} {1 - dist:7.3f}  {id_}")
