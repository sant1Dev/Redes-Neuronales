# Redes Neuronales

Implementaciones en Python de redes neuronales, desde un perceptrón simple hasta redes convolucionales.

## Contenido

| Carpeta | Qué hace | Librerías |
|---|---|---|
| [`perceptron_simple`](perceptron_simple) | Perceptrón con función signo (AND/XOR) | Ninguna |
| [`perceptronlineal_y_nolineal`](perceptronlineal_y_nolineal) | Perceptrón no lineal (tanh) con split 80/20 | `numpy`, `requests` |
| [`perceptron_multicapa`](perceptron_multicapa) | Red multicapa con backpropagation que resuelve XOR | Ninguna |
| [`autoencoders`](autoencoders) | Autoencoder de caracteres 7x5: espacio latente y eliminación de ruido | `tensorflow`, `matplotlib`, `numpy` |
| [`convolucionales`](convolucionales) | Clasificación de animales: CNN propia vs. MobileNetV2 (transfer learning) | `tensorflow`, `matplotlib`, `numpy` |

## Cómo ejecutar

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

## Notas

- **`perceptronlineal_y_nolineal`**: descarga los `.txt` desde una URL de GitHub, así que necesita internet. Para usar los archivos locales, se puede reemplazar la carga por `np.loadtxt("perceptronlineal_y_nolineal/conjunto_entrenamiento.txt")` y `np.loadtxt("perceptronlineal_y_nolineal/salida_deseada.txt")`.
- **`autoencoders`**: necesita un `fonts.txt`; la ruta al archivo se coloca en `ruta_archivo`, dentro de `cargar_datos()`. Por defecto corre el PUNTO 2 (eliminación de ruido); el PUNTO 1 está comentado.
- **`convolucionales`**: el dataset de fotos no está en el repositorio y hay que armarlo:
  1. Crear la carpeta `dataset/Animales/` en la raíz del repositorio.
  2. Dentro, crear una subcarpeta por animal con sus fotos (`.jpg` o `.png`). Sugeridas: `Gatos`, `Perros`, `Caballos`, `Aves`, `Conejos`, `Serpientes`, `Tortugas`, `Pinguinos`, `Hipopotamos`.
  3. Conviene tener varias fotos por animal (idealmente 50 o más) y una cantidad parecida en cada carpeta.

  El nombre de cada subcarpeta es el nombre de la clase. Las imágenes se pasan a 512x512 en grises (si falta memoria, se puede bajar `BATCH_SIZE`) y los historiales de entrenamiento se guardan en archivos `.csv`.
- Los hiperparámetros (`COTA`, `tasa_aprendizaje`, `beta`, `EPOCHS_*`) se ajustan al principio o al final de cada script.
