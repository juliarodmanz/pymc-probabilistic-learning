## Probabilistic ML Lab (PyMC)
Experimentos de modelado probabilístico y redes neuronales bayesianas con PyMC. Incluye los siguientes casos de estudio: 

* Predicción de desplazamiento de un robot con sensores ruidosos, comparación entre una BNN y una red determinista, y análisis de incertidumbre epistémica en contraposición a la aleatoria.

Objetivos
PONER

Estructura del repositorio
* src/
    * data/: generación de datasets sintéticos.
    * models/: BNN en PyMC y baseline determinista con PyTorch.
    * experiments/: scripts de entrenamiento y evaluación.
    * viz/: funciones de visualización.
    * utils/: semillas y métricas./
* reports/
    * figures/ (salida de gráficos), resultados.md, modelos guardados.
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
* Entrenar modelos
    python src/experiments/train_robot.py
    python src/experiments/train_biometrics.py
* Figuras en reports/figures/ (si las activas desde src/viz/plots.py)
* Descripción técnica (resumen)
    * Experimento 1: Cinemática del Robot (Regresión y Vacío Epistémico)
        * Preprocesamiento: Normalización Z-score aplicada exclusivamente a las entradas y salidas del modelo bayesiano para garantizar la convergencia del integrador numérico
        * BNN: arquitectura 1-50-50-1 con activaciones ReLU implementada en PyMC.
        * Priors: pesos y sesgos ~ N(0, 1).
        * Verosimilitud: y ~ N(f_θ(x), σ_obs), con σ_obs ~ HalfNormal.
        * Inferencia: ADVI (Inferencia Variacional Automática) o MCMC (NUTS).
        * Incertidumbre: 
            * Total: Var[y*|x*,D] estimada por muestras de la posterior.
            * Aleatoria: E[σ_obs^2] (homoscedástica en este caso base).
            * Epistémica: Var_total - E[σ_obs^2].
        * Baseline: Perceptrón Multicapa en PyTorch (50, 50), activación ReLU, optimización MSE (Adam).

    * Experimento 2: Seguridad Biométrica (Clasificación y Detección OOD)
        * Transformación Geométrica: Expansión polinómica de la entrada original para garantizar la generación de fronteras de decisión cerradas.
        * BNN: Arquitectura 4-20-3 con activación oculta ReLU y salida categórica Softmax, implementada en PyMC.
        * Priors:  Pesos de la capa oculta siguen una normal centrada en el origen con varianza  σ^2= 2/4, pesos de salida siguen una normal centrada en el origen con  σ^2= 1/20, y sesgos siguen una normal centrada en el origen con varianza 1.
        * Verosimilitud: Distribución Categórica
        * Inferencia: ADVI (Inferencia Variacional Automática) o MCMC (NUTS).
        * Incertidumbre: Cuantificada mediante la Entropía Predictiva de Shannon
        * Baseline: rceptrón Multicapa estocástico en PyTorch (4-32-3). Optimizado minimizando la Entropía Cruzada mediante Adam con regularización L2. Utiliza MCDropout mantenido activo durante la inferencia como mecanismo para estimar la incertidumbre.
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
