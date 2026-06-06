# Modelamiento Estocástico del Precio del Oro mediante Procesos de Difusión y Simulaciones Monte Carlo

Proyecto desarrollado para el curso **FIS205 - Física Computacional**.

Este repositorio contiene un pipeline computacional para analizar el precio histórico del oro desde una perspectiva de econofísica, utilizando retornos logarítmicos, estadística descriptiva, Movimiento Browniano Geométrico (GBM) y simulaciones Monte Carlo.

El objetivo principal no es construir un sistema de trading ni predecir precios de mercado con fines financieros, sino estudiar el comportamiento del precio del oro como un proceso estocástico, caracterizando sus fluctuaciones y comparando sus propiedades empíricas con un modelo difusivo simple.

---

## Autor

**Diego Yáñez Valdivia**

---

## Descripción general del proyecto

El proyecto estudia la evolución histórica del precio del oro, expresado en dólares estadounidenses por onza troy \((\mathrm{US\$}/\mathrm{ozt})\). A partir de esta serie temporal, se construyen retornos logarítmicos diarios y se analiza su comportamiento estadístico.

Posteriormente, se estiman los parámetros de un modelo de Movimiento Browniano Geométrico:

\[
dS_t = \mu S_t dt + \sigma S_t dW_t,
\]

donde:

- \(S_t\) representa el precio del oro;
- \(\mu\) representa la deriva del proceso;
- \(\sigma\) representa la volatilidad;
- \(W_t\) representa un proceso de Wiener.

Con los parámetros estimados desde los datos históricos, se generan trayectorias sintéticas mediante simulaciones Monte Carlo.

---

## Estado actual del proyecto

Hasta el avance 2, el pipeline incluye:

1. Preprocesamiento de datos históricos del precio del oro.
2. Construcción de retornos logarítmicos diarios.
3. Análisis estadístico de los retornos.
4. Comparación de la distribución empírica con una normal ajustada.
5. Análisis de autocorrelación temporal.
6. Estimación de parámetros del modelo GBM.
7. Simulación Monte Carlo de trayectorias del precio del oro.
8. Generación de gráficos y tablas de resultados.
9. Informe de avance 2 en formato PDF.

---

## Estructura del repositorio

```text
gold-price-stochastic-model/
│
├── README.md
├── preprocess_data.py
├── compute_returns.py
├── analyze_returns.py
├── simulate_gbm.py
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── gold_prices_clean.csv
│       └── gold_returns.csv
│
├── outputs/
│   ├── price_series.png
│   ├── log_returns_series.png
│   ├── log_returns_histogram.png
│   ├── log_returns_normal_fit.png
│   ├── returns_autocorrelation.png
│   ├── returns_statistics.csv
│   ├── gbm_parameters.csv
│   ├── gbm_monte_carlo_paths.png
│   ├── gbm_mean_path_with_band.png
│   ├── gbm_final_price_distribution.png
│   └── gbm_simulation_summary.csv
│
├── reports/
│   └── Avance_2_FIS205.pdf
│
└── notebooks/
```