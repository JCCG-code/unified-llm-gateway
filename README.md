# unified-llm-gateway

<p align="center">
    <em>Una aplicación de <b>FastAPI</b>, con modelos de <b>Llama3.2</b> y <b>Gemma4-e4b</b>, con un gran alto rendimiento potenciado por <b>uv</b></em>
</p>

## Qué es

Proyecto creado para una comunicación fácil y rápida con modelos locales ejecutándose en su ordenador. Ahora presta soporte para modelos de Llama3.2 y Gemma4-e4b. Fácil de usar y obtén respuestas ultrarrápidas.

## Stack

- **uv**: Gestor de paquetes y entornos Python moderno. Reemplaza pip/conda con mayor velocidad y reproducibilidad.
- **FastApi**: construye APIs en Python de forma fácil y consistente.
- **Pydantic v2.0**: construcción de modelos perfecta para la comunicación de SDKs de LLMs.
- **Pytest**: tests seguros y rápidos.
- **tiktoken**: cuenta tokens por parte de OpenAI.

## Endpoints

| Método | Ruta            | Descripción                                                                                                             |
| ------ | --------------- | ----------------------------------------------------------------------------------------------------------------------- |
| GET    | /health         | Comprueba el estado del servidor                                                                                        |
| POST   | /complete       | Llama a modelos `Llama3.2` o `Gemma4-e4b` y obtiene un `CompletionResponse`                                             |
| POST   | /complete/batch | Lo mismo que el anterior pero con llamadas concurrentes                                                                 |
| GET    | /models         | Obtiene todos los modelos disponibles                                                                                   |
| POST   | /models         | Crea un nuevo modelo                                                                                                    |
| GET    | /models/{name}  | Obtiene los datos de un modelo por nombre                                                                               |
| PUT    | /models/{name}  | Actualiza los datos de un modelo por nombre                                                                             |
| DELETE | /models/{name}  | Elimina un modelo por nombre                                                                                            |
| POST   | /estimate-cost  | Te permite saber un coste estimado del prompt                                                                           |
| POST   | /tokenize       | Tokeniza el prompt mediante la herramienta `tiktoken` de OpenAI                                                         |
| POST   | /compare        | A partir de un prompt, compara una tokenización con `tiktoken` y la orientada `// 4` y obtienes una estimación de error |

## Cómo ejecutarlo

Descarga el repositorio:

```bash
git clone https://github.com/JCCG-code/unified-llm-gateway.git
```

---

Instala las dependencias:

```bash
uv sync
```

---

Instala **Ollama** en tu máquina
macOS y Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Windows:

```bash
irm https://ollama.com/install.ps1 | iex
```

---

Descarga modelos
Llama3.2:

```bash
ollama pull llama3.2
```

Gemma4-e4b:

```bash
ollama pull batiai/gemma4-e4b:q4
```

---

Inicia el proyecto en modo desarrollador:

```bash
uv run fastapi dev src/main.py
```

## Decisiones técnicas

- **uv** sobre **pip**: gestión de dependencias reproducible y determinista. 10x más rápido que pip para instalar paquetes. Equivalente moderno a pnpm en el ecosistema Python.
- **Pydantic v2.0** sobre **dataclasses**: permite más versatilidad a la hora de crear modelos de datos, y poder trabajar con LLMs de forma más exacta y segura, controlando siempre los datos que entran y salen de la API en todo momento.
- **Llamadas Batch**: el uso de llamadas concurrentes mediante `asyncio.gather()` ha incrementando el rendimiento de las llamadas concurrentes a los LLMs y ha sido indispensable crear el modelo de trabajo.
- Uso de **tiktoken**: para controlar los tokens de entrada de manera real, el uso de la libreria `tiktoken` de OpenAI ha sido indispensable, para poder tener un número real calculado, que ni viene de la llamada al modelo ni tampoco de la división trivial `// 4` para el cálculo de tokens. Ganancia de umbral en estimación de costes.
- Uso de **fallback**: la llamada al modelo siempre. Aunque un modelo falle, automáticamente se activa el siguiente modelo disponible para proporcionar una respuesta al usuario. El resultado es la prioridad.

## Próximos pasos

1. Streaming real via SSE en /complete
2. Persistencia de modelos en Qdrant o PostgreSQL
3. Autenticación con API keys por usuario
4. Cost tracking por usuario con métricas reales
5. Soporte para más modelos Ollama
