# AI Text Detector (probabilístico y explicable)

Proyecto profesional en Python para estimar **probabilidad de uso de IA en textos** mediante señales lingüísticas, estructurales y semánticas.

> ⚠️ **Aviso clave**: este sistema **no es concluyente**. Produce una estimación probabilística y puede generar falsos positivos/falsos negativos.

---

## Qué hace

- Preprocesa texto: limpieza, idioma, tokenización por palabras/oraciones/párrafos.
- Extrae múltiples señales:
  - longitud media y desviación de oraciones,
  - burstiness,
  - TTR y frecuencia de palabras únicas,
  - repetición de n-gramas,
  - distribución de puntuación,
  - frecuencia de conectores,
  - similitud semántica entre párrafos (embeddings o fallback TF-IDF),
  - uniformidad estructural,
  - indicadores de voz genérica,
  - proxy de predictibilidad/perplexity.
- Ofrece 2 rutas de modelado:
  1. **Modelo manual interpretable** (features diseñadas).
  2. **Modelo híbrido** (features manuales + embeddings) calibrado.
- Interfaz web con:
  - score 0–100,
  - nivel de riesgo (Bajo/Medio/Alto),
  - tabla de features,
  - explicación interpretable,
  - análisis por oración y párrafo,
  - disclaimer visible legal/técnico.

## Qué no hace

- No determina autoría con certeza.
- No reemplaza revisión humana o forense.
- No garantiza robustez ante textos manipulados/adversariales.

## Limitaciones

- Dependencia del dataset: sesgos y dominio afectan resultados.
- Cambios en modelos generativos pueden degradar desempeño.
- Textos muy cortos tienen alta incertidumbre.
- El `perplexity_proxy` es aproximación estadística, no perplexity LLM “real”.

## Estructura

```text
/ai_detector
    /app
        main.py
        routes.py
        /templates
        /static
    /core
        preprocessing.py
        feature_extraction.py
        scoring.py
        explainability.py
        thresholds.py
    /models
        train_model.py
        inference.py
        /saved_model
    /utils
        text_helpers.py
        validators.py
        logging_config.py
    config.py
    requirements.txt
    README.md
```

## Instalación

```bash
cd ai_detector
python -m venv .venv
source .venv/bin/activate  # en Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Entrenamiento

Dataset esperado (`CSV`):

```csv
text,label
"Texto humano...",0
"Texto generado por IA...",1
```

Ejecuta:

```bash
python -m ai_detector.models.train_model --csv /ruta/dataset.csv --output-dir ai_detector/models/saved_model
```

Salida:
- `hybrid_calibrated_model.joblib`
- `manual_model.joblib`
- `training_metrics.json` (accuracy, precision, recall, f1, roc_auc, matriz de confusión)

## Ejecución de la app web

```bash
python -m ai_detector.app.main
```

Abre: `http://127.0.0.1:5000`

## Interpretación del puntaje

- `0–30`: **Bajo**
- `31–65`: **Medio**
- `66–100`: **Alto**

Estos umbrales son configurables en `config.py`.

## Filosofía de uso responsable

El sistema detecta **señales compatibles** con escritura asistida por IA. Úsalo como herramienta de apoyo para priorizar revisión humana, **nunca** como sentencia definitiva.

## Próximas extensiones recomendadas

- SHAP para explicabilidad de modelos entrenados.
- Versionado de datasets/modelos (DVC/MLflow).
- API auth + rate limiting.
- Soporte multilingüe robusto por idioma/registro.
