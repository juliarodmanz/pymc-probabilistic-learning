## Probabilistic ML Lab (PyMC)
Experimentos de modelado probabilístico y redes neuronales bayesianas con PyMC. Incluye los siguientes casos de estudio: 

* Predicción de desplazamiento de un robot con sensores ruidosos, comparación entre una BNN y una red determinista, y análisis de incertidumbre epistémica en contraposición a la aleatoria.

Objetivos
PONER

Estructura del repositorio
* src/
    * data/: generación de datasets sintéticos.
    * models/: BNN en PyMC y baseline determinista.
    * xperiments/: scripts de entrenamiento y evaluación.
    * viz/: funciones de visualización.
    * utils/: semillas y métricas.
* data/
    * raw/, processed/
* reports/
    * figures/ (salida de gráficos), resultados.md, modelos guardados.
* notebooks/
    * cuadernos de exploración.
* scripts/
    * setup_env.sh, run_all.sh.
* requirements.txt, LICENSE, README.md

Instalación rápida


Crear y activar entorno virtual

* Windows:
    python -m venv .venv
    .venv\Scripts\activate
* Linux/macOS:
    python -m venv .venv
    source .venv/bin/activate
* Instalar dependencias
    pip install --upgrade pip
    pip install -r requirements.txt
* Reproducir el experimento
* Generar datos sintéticos
    python src/data/make_dataset.py
* Entrenar modelos
    python src/experiments/run_training.py
* Guarda: reports/bnn_idata.nc y reports/models/mlp.joblib
* Evaluar y generar salidas
    python src/experiments/run_evaluation.py
* Métricas y resumen en reports/resultados.md
* Figuras en reports/figures/ (si las activas desde src/viz/plots.py)
* Descripción técnica (resumen)
    * BNN: arquitectura 1-50-50-1 con activaciones ReLU implementada en PyMC.
    * Priors: pesos y sesgos ~ N(0, 1).
    * Verosimilitud: y ~ N(f_θ(x), σ_obs), con σ_obs ~ HalfNormal.
    * Inferencia: ADVI (Inferencia Variacional Automática) o MCMC (NUTS).
    * Incertidumbre: 
    * Total: Var[y*|x*,D] estimada por muestras de la posterior.
    * Aleatoria: E[σ_obs^2] (homoscedástica en este caso base).
    * Epistémica: Var_total - E[σ_obs^2].
    * Baseline: Perceptrón Multicapa en PyTorch (50, 50), activación ReLU, optimización MSE (Adam).

* Resultados esperados
    * In-distribution (ID): RMSE bajo en ambos modelos, incertidumbre moderada.
    * Out-of-distribution (OOD) y fallos de sensor:
        * BNN incrementa la varianza epistémica drásticamente, señalando baja fiabilidad en la zona ciega.
        * Red clásica produce una predicción puntual sobreconfiada sin indicador de error.

* Cómo adaptar/expandir
    * Heteroscedasticidad: modelar σ_obs(x) con una segunda "cabeza" de salida.
    * Más capas/ancho: editar los parámetros en `src/models/mlp_determinista.py` y `src/models/modelon_bayesiano.py`.
    * Datos reales: coloca el dataset en `data/raw` y modifica `src/experiments/train_robot.py`.

* Reproducibilidad
    * Semillas fijadas en `utils/seeds.py`.
    * Versiones en `requirements.txt`. Para congelar el entorno exacto:
        pip freeze > requirements-lock.txt

Licencia
Código bajo licencia MIT (ver LICENSE).


Julia Rodríguez Manzanedo (2026). Probabilistic ML Lab (PyMC). https://github.com/juliarodmanz/pymc-probabilistic-learning.git.
Contacto
Autora: Julia Rodríguez Manzanedo
Email: juliarodmanz@gmail.com
