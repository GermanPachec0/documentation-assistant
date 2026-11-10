## Paso 0 — Entorno y fundamentos de LLMs

- Proyecto manejado con `uv`: `uv sync` instala las dependencias, `uv add <paquete>` agrega una nueva y `uv run python archivo.py` ejecuta usando el entorno del proyecto.
- El nombre con el que se instala un paquete no siempre es el nombre con el que se importa: `python-dotenv` se importa como `dotenv` y `google-genai` como `from google import genai`.
- VS Code tiene que usar el intérprete de `.venv` para que funcione el autocompletado.
- Primer llamado a Gemini: `load_dotenv()` carga la API key, `genai.Client()` crea el cliente, y `client.models.list()` devuelve un "pager" que hay que recorrer con un `for`.

## Paso 1 — Exploración de documentos y modelo de fragmento

- Con `page.get_text("dict")`, PyMuPDF da la estructura del PDF en tres niveles: bloques, líneas y spans. Cada span tiene tamaño, fuente, color y texto.
- En este PDF lo que distingue a cada elemento es el **color**, no el tamaño:
  - Encabezado y pie: color `#1c1482`.
  - Títulos de endpoint (`HTTP request:`): `#0070c0` o `#2f5496`.
  - Código: Consolas, con `public` en azul y las descripciones en verde.
- Los índices de página de PyMuPDF arrancan en 0: el índice 48 es la página 49 del visor y la "48" impresa en el pie.
- Definí el modelo `Fragmento` con Pydantic, en `src/models.py`. Pydantic valida los campos: si falta uno obligatorio, tira un error.
- Chroma solo acepta metadatos simples (str, int, float, bool). Por eso los parámetros van dentro del `texto` y no como una lista aparte.

## Paso 2 — Lector del PDF de ESCO y chunking

### Experimento de chunking
Probé tres formas de cortar el texto: cada 500 caracteres, cada 2000 y una vez por endpoint. Los cortes fijos parten los endpoints por la mitad y dejan pedazos que no dicen de qué endpoint son. Cortar por endpoint deja cada uno completo. La primera métrica que usé engañaba: pedía solo tres palabras, que aparecían al principio, así que hubo que exigir también el último campo del endpoint.

### Piezas del lector (`src/readers/pdf_esco.py`)
- **`parsear_titulo`**: saca el método y el nombre del endpoint. Tuvo que tolerar `->` o `-`, guiones largos, espacios en medio del nombre y puntos finales. Si no reconoce el título, tira un error en lugar de saltearlo sin avisar.
- **`es_ruido`**: descarta el encabezado, el pie y el número de página. El número de página es negro como el texto normal, así que se reconoce por su **posición**: medí que está en y0/alto = 0.897, y el contenido real llega hasta 0.880. La regla es "está abajo **y** es solo dígitos": con una sola de las dos condiciones se borraría contenido.
- **`leer_pdf`**: recorre las páginas línea por línea con una "máquina de estados" (siempre hay como mucho un endpoint abierto).

### Problemas que encontré y cómo los resolví
- **El color del título:** solo 20 de los 198 títulos tenían el color que esperaba. El criterio que funciona es "dice `HTTP request:` y no es negro". Los negros son el índice impreso.
- **Una viñeta parecía un título:** `● HTTP request: DELETE…` era un ítem de una lista, no un endpoint.
- **IDs repetidos:** `domicilio-cuenta-comitente` existe como GET, DELETE y PUT, así que el ID pasó a incluir el método: `esco:get:domicilio-cuenta-comitente`.
- **Endpoints documentados dos veces y con contradicciones:** los decidí a mano y dejé el motivo escrito en el código, en la lista `DESCARTAR`. Por ejemplo, en `get-tipos-contrib-retencion` una versión dice `long`, pero los ejemplos son textos ("EX"), así que la correcta es la que dice `string`. Si aparece un duplicado nuevo, el lector tira un error en lugar de elegir solo.
- **La descripción venía ANTES del título:** pasaba en unos 120 endpoints. La descripción se perdía o quedaba pegada al endpoint anterior. Lo resolví con el **preámbulo**: el subtítulo cierra el endpoint anterior, sus líneas se guardan aparte y se ponen al principio del endpoint siguiente.
- **Dos niveles de encabezado:** la **sección** (azul oscuro, negrita) es el grupo y va al campo `seccion`. El **subtítulo** ("Cancelar Orden") va dentro del texto, porque se parece a lo que pregunta un usuario.
- **Notas de negocio:** "Validación de FCI para operar" y "Órdenes a precio de mercado" no son endpoints, pero tienen información útil. Ahora son fragmentos propios con `tipo="nota"`.
- **Endpoints casi vacíos** (`delete-condominio`, `persona-hist`, `update-condominio`): revisé que no hubiera nada oculto. El PDF realmente no tiene más información sobre ellos.

### Resultado
166 fragmentos (164 endpoints y 2 notas) guardados en `data/chunks/esco.json`.

### Respuestas de cierre
1. **El preámbulo:** antes, el fragmento nuevo se abría recién en `HTTP request:`, y el subtítulo con su descripción, que están antes, quedaban en el fragmento anterior. Eso causaba dos daños: al endpoint anterior le **sobraba** texto que no era suyo, y al nuevo le **faltaba** la frase que dice qué hace, que es la que más se parece a la pregunta de un usuario. El preámbulo guarda esas líneas y se las da al endpoint correcto.
2. **El fragmento gigante:** `insert-cuenta-comitente` tiene unos 24.000 caracteres (unos 6.000 tokens). El modelo de embeddings solo lee el principio, así que más del 90 % no se tendría en cuenta al buscar. Cortarlo cada 500 caracteres dejaría pedazos que no dicen de qué endpoint son. La solución, que queda para el Paso 4, es dividir solo los fragmentos grandes y repetir el encabezado en cada parte.
3. **¿Dónde está la falla?** La información sobre los FCI no estaba en ningún fragmento, así que la falla era de **chunking**: la búsqueda no puede encontrar lo que no existe. Para distinguirlo, se revisa en orden: ¿existe un fragmento con la respuesta? (chunking), ¿la búsqueda lo trajo? (recuperación), ¿el LLM lo usó bien? (generación). El primer eslabón que falla es el culpable.


## Paso 4 — Ingesta: embeddings y similitud

### Qué es un embedding
Una lista de números (384 en los modelos que probé) que representa el significado de un texto. Textos que significan lo mismo quedan "cerca". La cercanía se mide con la **similitud coseno**: si los vectores están normalizados (miden 1), el coseno es solo el producto punto.

### Ejercicio de similitud (`scratch/similitud.py`)
- **La negación no cuenta:** "no actualizar el domicilio" fue la frase MÁS parecida a "actualizar el domicilio" (0.88 en MiniLM, 0.98 en e5). El modelo ve el tema, no si dice sí o no.
- **Los typos casi no afectan** (0.82), pero los **sinónimos sí** (0.61): pesan mucho las palabras compartidas.
- **Las escalas no se comparan entre modelos:** en e5, hasta "receta de empanadas" da 0.79. Lo que se compara es el orden y el margen.

### Elegir el modelo con datos: MiniLM contra e5-small
| | MiniLM | e5-small |
|---|---|---|
| Tokens que lee como máximo | 128 | 512 |
| Fragmentos que superan el límite | 81 % | 30 % |
| Texto que el modelo no lee | 80 % | 49 % |

La prueba de frases favorecía a MiniLM, pero los tokens dicen que habría ignorado casi todo. Elegí **e5-small**, y lo decidí midiendo los datos, sin probar ni una pregunta.

### Partir solo los fragmentos largos (`src/chunking.py`)
- Solo se parten los que superan 500 tokens: 166 padres pasan a ser 317 hijos.
- Se corta antes de un parámetro (`public`), de `Request:`, de `Response:` o de `Ejemplo`, nunca en el medio.
- Cada parte repite el encabezado del padre y, si arranca en medio de una tabla, lleva la marca `[Response, continúa]`.
- El id del hijo es `padre#n`, y el padre se recupera con `split("#")[0]`.
- Un chequeo que no puede fallar no verifica nada: comparar conjuntos de líneas no detectaba una `}` perdida. El chequeo estricto reconstruye el cuerpo del padre en orden.

### Ingesta en Chroma (`src/ingest.py`)
- `upsert` y no `add`: correr la ingesta dos veces deja lo mismo. Además, borra los ids que ya no existen.
- Chroma no acepta `None`: las notas usan `metodo=""` y las fuentes sin páginas `0`.
- `get_or_create_collection` abre una colección vieja con su configuración vieja: hay que verificar que el espacio sea `cosine`.

### El hallazgo más importante
"¿Cómo **actualizo** el domicilio?" trajo primero el **DELETE** (el que borra), y el PUT quedó 9.º. El embedding ve el tema (domicilio) y no la acción. Con solo el encabezado (método, subtítulo y descripción), el PUT ganaba: lo que lo hundía era el cuerpo, lleno de parámetros parecidos en los tres endpoints.

## Paso 5 — Recuperación: BM25, búsqueda vectorial y RRF

### La evaluación (`eval/`)
- 15 preguntas escritas por mí en `eval/preguntas.yaml`, con categorías: acción, sinónimo, técnica, nota, gigante, inglés, typos, ambigua y sin respuesta.
- Métricas: **hit@1** (el primero es el correcto), **hit@5** (el correcto está entre los 5 primeros) y **MRR** (qué tan arriba aparece).
- Con 13 preguntas evaluables, una sola pregunta mueve hit@5 en 0.08: hay que mirar qué preguntas cambian, no solo el promedio.

### Estrategias (todas con la misma firma: `buscar(pregunta, k) -> padre_ids`)
| Estrategia | hit@1 | hit@5 | MRR |
|---|---|---|---|
| Semántica (línea base) | 0.31 | 0.62 | 0.41 |
| BM25 (palabras exactas) | 0.23 | 0.46 | 0.33 |
| Híbrida (RRF) | 0.38 | 0.54 | 0.44 |
| Semántica + encabezado + verbos | 0.38 | **0.69** | **0.53** |
| Híbrida + encabezado + verbos | **0.46** | 0.62 | 0.50 |

- **BM25** no arregló ninguna falla: el verbo del usuario ("actualizo", "quitar") no aparece en ningún fragmento.
- **RRF** combina por posición, no por puntaje. Ayuda cuando las dos búsquedas fallan en preguntas distintas, pero no puede encontrar lo que ninguna trae.
- **Encabezado como vector aparte:** arregló el caso del domicilio (de no aparecer al 1.º puesto).
- **Verbos en la primera línea:** en este PDF el método no dice la acción (37 consultas usan POST), así que los verbos salen del prefijo del nombre (`insert-` → crear, `get-` → consultar).
- **Glosario de negocio** (`data/glosario.yaml`): "acreditar/debitar dinero" funciona en el 1.º puesto, pero "agregar/quitar saldo" no. El glosario solo ayuda con las palabras que tiene.

### Lo que sigue fallando
- **Typos con acción** ("actualiso el domisilio"): candidato para un reranker.
- **Vocabulario de negocio** ("saldo"): solo se arregla con el glosario.
- **Preguntas ambiguas** ("operaciones de un cliente"): no se arreglan en la búsqueda; el LLM tiene que preguntar.

## Paso 6 — LLM y generación

### Cómo funciona
Pregunta → búsqueda (5 endpoints) → contexto etiquetado → Gemini → respuesta con citas.
- `src/llm.py` es la única puerta al LLM: si cambio de proveedor, toco solo ese archivo. Reintenta solo los errores 429 y 503 (esperando 2, 4 y 8 segundos) y devuelve tokens y tiempo.
- `src/generation.py` arma el prompt: las reglas van en el SISTEMA y los fragmentos y la pregunta en el mensaje.

### Las reglas del SISTEMA, y por qué
1. Responder solo con las fuentes (si no, mezcla lo que "recuerda", como el ESCO europeo).
2. Si no hay nada, una frase fija: "No encontré ningún endpoint para eso…". Hay que darle permiso explícito para no saber.
3. Si el endpoint existe pero no está detallado, otra frase distinta. Antes las dos reglas se pisaban y el modelo se contradecía.
4. Citar con [FUENTE n].
5. Elegir el endpoint por la acción (PUT contra DELETE).
6. Si la pregunta es ambigua, preguntar en lugar de elegir al azar.
7. Mostrar un ejemplo de request solo si el endpoint trae uno.

### Problemas que encontré y cómo los resolví
- **Costo:** los endpoints gigantes iban completos. Solución: mandar el hijo más parecido a la pregunta y sus dos vecinos.

| Pregunta | Tokens antes | Tokens después | Ahorro |
|---|---|---|---|
| Actualizar domicilio | 10.592 | 3.335 | −69 % |
| Parámetros de update-condominio | 12.631 | 4.500 | −64 % |
| Receta de pan casero | 15.003 | 7.124 | −53 % |
| FCI para operar | 2.603 | 2.699 | (sin gigantes) |

- **Citas mezcladas:** el modelo copiaba mal las etiquetas (el nombre de una fuente con las páginas de otra). Solución: el modelo cita [FUENTE n] y Python arma la etiqueta. **El LLM decide, el código formatea.**
- **Ejemplo inventado:** el PUT de domicilio no tiene ejemplo en el PDF, y el modelo armó uno con valores de otros endpoints y lo presentó como "de la documentación". Lo encontré verificando a mano cada dato de la respuesta (fidelidad). Solución: la regla 7.
- **Pendiente:** el número de cita ya no se equivoca de formato, pero el modelo todavía puede elegir el número equivocado. Eso se mide en el Paso 8.

### Respuestas de cierre
1. **Citas por número:** copiar etiquetas largas es mecánico, y el LLM las mezcla porque genera texto palabra por palabra. Un número es difícil de equivocar, y Python formatea sin errores y puede verificar que la fuente exista. Principio: el LLM decide, el código formatea.
2. **Pan casero:** lo hizo posible el prompt (regla 2), no la búsqueda: la búsqueda siempre devuelve 5 resultados, y con e5 hasta lo irrelevante tiene puntaje alto, así que no sirve un umbral. Sin la regla 2, el modelo tendería a dar la receta de memoria o a forzar los fragmentos, porque está entrenado para ser útil.

## Paso 7 — Interfaz

## Paso 8 — Evaluación

## Paso 9 — Publicación
