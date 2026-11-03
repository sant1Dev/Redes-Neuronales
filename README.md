<h1 align="center">🧠 Redes Neuronales</h1>

Repositorio con implementaciones de redes neuronales en Python, ordenadas de menor a mayor complejidad. Sirve como recorrido por los fundamentos del aprendizaje automático: se empieza con un perceptrón que aprende una sola neurona a mano, se continúa con redes multicapa entrenadas con backpropagation escrita desde cero, y se termina con autoencoders y redes convolucionales construidos con TensorFlow/Keras.

Las primeras carpetas están hechas en Python puro, sin librerías de aprendizaje automático, para ver cómo funciona cada paso del algoritmo: producto interno, función de activación, cálculo del error y actualización de pesos. Las últimas usan TensorFlow/Keras para trabajar con imágenes y datasets más grandes.

## 📚 Contenido

| Carpeta | Qué hace | Librerías |
|---|---|---|
| [`perceptron_simple`](perceptron_simple) | Perceptrón con función signo (AND/XOR) | Ninguna |
| [`perceptronlineal_y_nolineal`](perceptronlineal_y_nolineal) | Perceptrón no lineal (tanh) con split 80/20 | `numpy`, `requests` |
| [`perceptron_multicapa`](perceptron_multicapa) | Red multicapa con backpropagation que resuelve XOR | Ninguna |
| [`autoencoders`](autoencoders) | Autoencoder de caracteres 7x5: espacio latente y eliminación de ruido | `tensorflow`, `matplotlib`, `numpy` |
| [`convolucionales`](convolucionales) | Clasificación de animales: CNN propia vs. MobileNetV2 (transfer learning) | `tensorflow`, `matplotlib`, `numpy` |

## 🔍 Detalle de cada carpeta

### 1. `perceptron_simple`
Perceptrón de una sola neurona con función de activación signo (salida `1` o `-1`), basado en el algoritmo clásico de aprendizaje por corrección de error. En cada iteración se elige un ejemplo al azar, se calcula la salida y se ajustan los pesos según el error. Se guarda el mejor conjunto de pesos encontrado (`w_min`).
- `perceptron_simple.py`: versión con funciones.
- `perceptron_simple_v2.py`: misma lógica, versión con clase.
- Entrena con la función **AND**. También se puede probar **XOR**, que un perceptrón simple no logra resolver porque no es linealmente separable.

### 2. `perceptronlineal_y_nolineal`
Perceptrón con activación **tanh** (no lineal) entrenado con un conjunto de 200 ejemplos de 3 entradas cada uno, incluidos en la carpeta (`conjunto_entrenamiento.txt` y `salida_deseada.txt`).
- Normaliza las salidas deseadas al intervalo [-1, 1] para poder usar `tanh`.
- Entrena con todos los datos y luego con una partición **80/20** (entrenamiento/test), mostrando las predicciones sobre los ejemplos de test.
- `perceptron_no_lineal.py`: versión con clase. `perceptron_nolineal_sinclase.py`: versión con funciones.
- Los comentarios del código incluyen una reflexión sobre validación cruzada (k-fold) para elegir el mejor conjunto de entrenamiento.

### 3. `perceptron_multicapa`
Red neuronal multicapa escrita desde cero, con **backpropagation** y descenso de gradiente por lotes (batch). Resuelve el problema **XOR**, imposible para el perceptrón simple.
- Arquitectura: 2 entradas, 3 neuronas ocultas y 1 salida, todas con activación `tanh`.
- Los pesos se inicializan al azar y se actualizan una vez por época con el gradiente promedio.
- El entrenamiento termina cuando el error baja de 0.001 o se alcanza el máximo de épocas, y se imprime la predicción para cada entrada.

### 4. `autoencoders`
Autoencoder con Keras que aprende a comprimir y reconstruir caracteres de 7x5 píxeles, cargados desde un archivo `fonts.txt` en formato hexadecimal (32 caracteres por bloque: símbolos y dígitos, mayúsculas o minúsculas).
- **Arquitecturas**: prueba de distintas redes simétricas con cuello de botella.
- **Espacio latente 2D**: comprime cada carácter a 2 números, grafica los 32 caracteres como puntos y permite **generar caracteres nuevos** a partir de coordenadas inventadas.
- **Eliminación de ruido**: entrena con copias ruidosas de cada carácter (píxeles invertidos al azar) y compara original, versión con ruido y reconstrucción.

### 5. `convolucionales`
Clasificador de imágenes de animales que compara dos enfoques sobre el mismo dataset (80% entrenamiento, 20% validación, imágenes de 512x512 en grises):
- **CNN propia**: tres bloques convolucionales con max pooling, capa densa y dropout.
- **MobileNetV2 con transfer learning**: primero se entrena solo la cabeza con la base congelada y después se hace *fine tuning* de las últimas 30 capas con una tasa de aprendizaje menor.
- Al final compara las predicciones de ambos modelos sobre imágenes sueltas y guarda los historiales de entrenamiento en archivos `.csv`, listos para graficar.

## ▶️ Cómo ejecutar

Requiere Python 3.9+. Se clona el repositorio y cada script se ejecuta desde la raíz:

```bash
git clone https://github.com/sant1Dev/Redes-Neuronales.git
cd Redes-Neuronales

python perceptron_simple/perceptron_simple.py        # para probar XOR, se cambia "AND" por "XOR" al final del archivo
python perceptron_multicapa/perceptron_mc.py

pip install numpy requests
python perceptronlineal_y_nolineal/perceptron_no_lineal.py

pip install tensorflow matplotlib numpy
python autoencoders/ae_caracteres.py
python convolucionales/conv_animales.py
```

## 📝 Notas

- **`perceptronlineal_y_nolineal`**: descarga los `.txt` desde una URL de GitHub, así que necesita internet. Para usar los archivos locales, se puede reemplazar la carga por `np.loadtxt("perceptronlineal_y_nolineal/conjunto_entrenamiento.txt")` y `np.loadtxt("perceptronlineal_y_nolineal/salida_deseada.txt")`.
- **`autoencoders`**: necesita un `fonts.txt`; la ruta al archivo se coloca en `ruta_archivo`, dentro de `cargar_datos()`. Por defecto corre el PUNTO 2 (eliminación de ruido); el PUNTO 1 está comentado.
- **`convolucionales`**: el dataset de fotos no está en el repositorio y hay que armarlo:
  1. Crear la carpeta `dataset/Animales/` en la raíz del repositorio.
  2. Dentro, crear una subcarpeta por animal con sus fotos (`.jpg` o `.png`). Sugeridas: `Gatos`, `Perros`, `Caballos`, `Aves`, `Conejos`, `Serpientes`, `Tortugas`, `Pinguinos`, `Hipopotamos`.
  3. Conviene tener varias fotos por animal (idealmente 50 o más) y una cantidad parecida en cada carpeta.

  El nombre de cada subcarpeta es el nombre de la clase. Si falta memoria, se puede bajar `BATCH_SIZE`.
- Los hiperparámetros (`COTA`, `tasa_aprendizaje`, `beta`, `EPOCHS_*`) se ajustan al principio o al final de cada script.
