## Probabilistic ML Lab (PyMC)
Experimentos de modelado probabilístico y redes neuronales bayesianas con PyMC. Incluye los siguientes casos de estudio: 

* Predicción de desplazamiento de un robot con sensores ruidosos, comparación entre una BNN y una red determinista, y análisis de incertidumbre epistémica en contraposición a la aleatoria.
* Detección de intrusos en sistemas biométricos de dispositivos móviles, comparación entre una BNN y una red determinista mejorada con MC Dropout.
* Predicción de demanda en sistemas de alquiler de bicicletas, comparación entre dos redes neuronales una entrenada con ADVI y otra con MCMC.

Objetivo:
El objetivo principal es estudiar, implementar y validar el paradigma de las Redes Neuronales Bayesianas como solución para la cuantificación de la incertidumbre predictiva en modelos de aprendizaje automático, demostrando empíricamente su superioridad frente a los modelos deterministas tradicionales en escenarios de datos limitados o fuera de distribución. También se compararán los motores de inferencia ADVI y MCMC frente a distintos volúmenes de datos.

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
    python src/experiments/train_bikes.py
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
     * Experimento 3:  Predicción de Demanda en Sistemas de Alquiler de Bicicletas
        * Preprocesamiento: Eliminación de índices y de las columnas de subclasificación de usuarios, ya que su suma equivale a la variable objetivo, evitando así una tautología predictiva. Posteriormente, se aplica una tipificación escalar Z-score tanto a las covariables continuas como al vector objetivo para acotar la magnitud de los gradientes.
        * BNN: Arquitectura topológica base con una capa oculta de 20 neuronas y el uso de la función tangente hiperbólica como función de activación.
        * Priors: Distribuciones gaussianas relajadas para las matrices de pesos y sesgos, con el fin de evitar una regularización temprana inducida por la Divergencia KL.
        * Verosimilitud: La incertidumbre aleatoria se modela mediante una distribución semigaussiana
        * Inferencia: Análisis comparativo del comportamiento algorítmico entre MCMC (Inferencia Exacta usando pm.Data ) y ADVI (Inferencia Variacional con aproximación de Campo Medio, iteradores estocásticos pm.Minibatch y optimizador Adam con una tasa de aprendizaje $\alpha=0.001$ ).
        * Métricas de Evaluación: Raíz del Error Cuadrático Medio, Probabilidad de Cobertura de Intervalos de Predicción y Varianza Predictiva Media.
        * Fases de Experimentación:
             * Fase 1: Evaluada sobre 731 observaciones y 11 características (day.csv). El motor MCMC demuestra alta nitidez predictiva con un RMSE de 649.05 y un PICP bien calibrado del 93.20%. En contraste, debido a los gradientes ruidosos en una muestra diminuta, el modelo con ADVI infla masivamente sus varianzas.
             * Fase 2: Escalada a 17,379 observaciones y 12 características (hour.csv). Al incrementar la volumetría, se demuestra la escalabilidad de ADVI, el cual logra estabilizar el ELBO y mejorar drásticamente su precisión puntual.
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
