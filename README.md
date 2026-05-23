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
Windows:
python -m venv .venv
.venv\Scripts\activate
Linux/macOS:
python -m venv .venv
source .venv/bin/activate
Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt
Reproducir el experimento
Generar datos sintéticos
python src/data/make_dataset.py
Entrenar modelos
python src/experiments/run_training.py
Guarda: reports/bnn_idata.nc y reports/models/mlp.joblib
Evaluar y generar salidas
python src/experiments/run_evaluation.py
Métricas y resumen en reports/resultados.md
Figuras en reports/figures/ (si las activas desde src/viz/plots.py)
Descripción técnica (resumen)
BNN: arquitectura 4–16–16–1 con activaciones tanh.
Priors: pesos y sesgos ~ N(0, σ_w^2) con σ_w≈1.
Verosimilitud: y ~ N(fθ(x), σ_obs), con σ_obs ~ HalfNormal.
Inferencia: NUTS (o ADVI si es necesario por tiempo).
Incertidumbre:
Total: Var[y*|x*,D] estimada por muestras de la posterior.
Aleatoria: E[σ_obs^2] (homoscedástica en este ejemplo).
Epistémica: Var_total − E[σ_obs^2].
Baseline: MLPRegressor (16,16), activación tanh, MSE.
Resultados esperados
In-distribution (ID): RMSE bajo en ambos, incertidumbre moderada.
Out-of-distribution (OOD) y fallos de sensor:
BNN incrementa varianza epistémica señalando baja fiabilidad.
Red clásica produce predicción puntual sin indicador de confianza.
Cómo adaptar/expandir
Heteroscedasticidad: modelar σ_obs(x) con una segunda “cabeza” y softplus.
Más capas/ancho: editar src/models/bnn_pymc.py.
Datos reales: coloca CSV/NPZ en data/raw y modifica src/data/make_dataset.py.
Reproducibilidad
Semillas fijadas en utils/seeds.py.
Versiones en requirements.txt. Para congelar el entorno:
pip freeze > requirements-lock.txt
Licencia
Código bajo licencia MIT (ver LICENSE).


Julia Rodríguez Manzanedo (2026). Probabilistic ML Lab (PyMC). https://github.com/juliarodmanz/pymc-probabilistic-learning.git.
Contacto
Autora: Julia Rodríguez Manzanedo
Email: juliarodmanz@gmail.com
