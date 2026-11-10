from typing import Literal

from pydantic import BaseModel, model_validator


class Fragmento(BaseModel):
    """Una unidad de la documentación, lista para embeber y guardar en Chroma.

    Hay dos tipos:
    - "endpoint": un HTTP request documentado (lo normal).
    - "nota": una regla de negocio que no es un endpoint pero aplica a varios
      (ej. "Validación de FCI para operar"). Se recupera sola, cuando es relevante.

    Todos los campos son tipos simples (str/int) porque Chroma solo acepta
    metadatos str, int, float o bool. Los parámetros van dentro de `texto`.
    Los campos que pueden ser None (metodo, páginas) hay que convertirlos al
    guardar en Chroma.
    """

    id: str  # "esco:post:get-condominios" | "esco:nota:validacion-de-fci-para-operar"
    tipo: Literal["endpoint", "nota"] = "endpoint"
    sistema: str  # "esco"
    endpoint: str  # endpoint: "get-condominios" | nota: slug del título
    metodo: str | None = None  # "POST"; None en las notas
    seccion: str  # agrupador del índice del documento
    texto: str  # lo que se embebe, sin encabezados ni pies
    fuente: str  # nombre del PDF o URL de la spec
    pagina_inicio: int | None = None  # None para fuentes OpenAPI
    pagina_fin: int | None = None

    @model_validator(mode="after")
    def _metodo_segun_tipo(self) -> "Fragmento":
        if self.tipo == "endpoint" and not self.metodo:
            raise ValueError("un fragmento de tipo 'endpoint' necesita metodo")
        if self.tipo == "nota" and self.metodo is not None:
            raise ValueError("una nota no tiene metodo")
        return self
