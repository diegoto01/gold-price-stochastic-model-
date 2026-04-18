# Modelamiento Estocástico del Precio del Oro

Proyecto de Física Computacional (FIS-205, UTFSM)

## Descripción
Este proyecto implementa un pipeline computacional para analizar el comportamiento del precio del oro como un sistema estocástico, utilizando herramientas de econofísica y simulación numérica.

El enfoque principal consiste en estudiar la serie temporal de precios reales, transformarla en retornos logarítmicos y, a partir de ello, evaluar su comportamiento probabilístico como paso previo a la implementación de modelos de difusión.

## Objetivos
- Analizar estadísticamente la serie de retornos del oro
- Evaluar si el comportamiento es consistente con un proceso difusivo
- Modelar el precio mediante Geometric Brownian Motion (GBM)
- Generar simulaciones Monte Carlo
- Comparar resultados con datos reales

## Estructura del proyecto
- `data/`
  - `raw/`: datos originales en formato Excel
  - `processed/`: datos limpios y series derivadas
- `src/`: scripts principales del pipeline
- `notebooks/`: exploración y visualización
- `outputs/`: gráficos y resultados

## Metodología
1. Preprocesamiento de datos
2. Cálculo de retornos logarítmicos
3. Análisis estadístico de la serie
4. Modelamiento mediante GBM
5. Simulación Monte Carlo
6. Comparación con datos reales

## Flujo de trabajo

El pipeline implementado actualmente cubre las siguientes etapas:

1. **Datos crudos (`data/raw/`)**
   - Archivos Excel descargados desde el Banco Central de Chile
   - Contienen precios diarios del oro

2. **Preprocesamiento (`preprocess_data.py`)**
   - Lectura de archivos Excel
   - Limpieza de datos (formato de fechas, valores faltantes)
   - Unificación en una sola serie temporal
   - Exportación a `data/processed/gold_prices_clean.csv`

3. **Cálculo de retornos (`compute_returns.py`)**
   - Cálculo de retornos logarítmicos:
     r_t = log(P_t / P_{t-1})
   - Generación de la serie de retornos
   - Exportación a `data/processed/gold_returns.csv`

---

## Etapas futuras

Las siguientes fases están contempladas en el proyecto, pero aún no han sido implementadas:

- Análisis estadístico completo de retornos
- Modelamiento mediante Geometric Brownian Motion (GBM)
- Simulación Monte Carlo
- Comparación entre simulaciones y datos reales
- Visualización avanzada de resultados

## Ejecución

Orden recomendado de ejecución:

```bash
python preprocess_data.py
python compute_returns.py
