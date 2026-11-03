import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.preprocessing import image

# parametros
IMG_HEIGHT = 512
IMG_WIDTH = 512
BATCH_SIZE = 16   # con 32 me daba OOM por el tamaño de las imagenes
EPOCHS_CUSTOM = 30
EPOCHS_TL_FASE1 = 1
EPOCHS_TL_FINE = 1
LEARNING_RATE = 0.001

# carpeta del dataset, una subcarpeta por clase (Gatos/, Perros/, etc.)
# la ruta es relativa al repo, cambiala si lo tenes en otro lado
DATA_DIR = os.path.join("dataset", "Animales")

# carga de datos en grises
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    color_mode='grayscale'
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    color_mode='grayscale'
)

class_names = train_dataset.class_names
num_clases = len(class_names)
print("Clases:", num_clases, class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)


# modelo 1: CNN propia
def build_custom_cnn():
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_clases, activation='softmax')
    ])
    return model

print("\nEntrenando CNN custom")
model_custom = build_custom_cnn()
model_custom.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

log_custom = CSVLogger('historial_custom.csv', append=False)
history_custom = model_custom.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS_CUSTOM,
    callbacks=[log_custom]
)


# modelo 2: MobileNetV2 con transfer learning
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False


# MobileNet espera 3 canales y yo tengo 1, asi que lo repito 3 veces
class RepeatChannels(layers.Layer):
    def __init__(self, repeats, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.repeats = repeats
        self.axis = axis

    def call(self, inputs):
        return tf.repeat(inputs, repeats=self.repeats, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"repeats": self.repeats, "axis": self.axis})
        return config


inputs_tl = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
x = RepeatChannels(repeats=3, axis=-1)(inputs_tl)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs_tl = layers.Dense(num_clases, activation='softmax')(x)

model_tl = tf.keras.Model(inputs_tl, outputs_tl)
model_tl.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTransfer learning, fase 1 (base congelada)")
log_tl1 = CSVLogger('historial_tl_fase1.csv', append=False)
history_tl_fase1 = model_tl.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS_TL_FASE1,
    callbacks=[log_tl1]
)

print("\nFine tuning")
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 30  # solo las ultimas 30 capas
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# lr mas chico para no romper los pesos de imagenet
model_tl.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

log_tl2 = CSVLogger('historial_tl_finetuning.csv', append=False)
history_tl_fine = model_tl.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS_TL_FINE,
    callbacks=[log_tl2]
)


# comparacion sobre imagenes sueltas
def predecir_y_comparar(ruta, model_c, model_t, classes, h, w):
    img = image.load_img(ruta, target_size=(h, w), color_mode='grayscale')
    arr = image.img_to_array(img)
    batch = np.expand_dims(arr, axis=0)

    p_c = model_c.predict(batch, verbose=0)[0]
    p_t = model_t.predict(batch, verbose=0)[0]

    clase_c, conf_c = classes[np.argmax(p_c)], 100 * np.max(p_c)
    clase_t, conf_t = classes[np.argmax(p_t)], 100 * np.max(p_t)

    plt.figure(figsize=(6, 5))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f"Custom: {clase_c} ({conf_c:.1f}%)\n"
              f"Transfer: {clase_t} ({conf_t:.1f}%)",
              pad=15, fontsize=12, fontweight='bold')
    plt.show()


# imagenes de ejemplo, ponele las que tengas en tu carpeta
base = os.path.join("dataset", "Animales")
imagenes_prueba = [
    os.path.join(base, "Conejos", "conejo_01.jpg"),
    os.path.join(base, "Serpientes", "serpiente_01.jpg"),
    os.path.join(base, "Tortugas", "tortuga_01.jpg"),
    os.path.join(base, "Gatos", "gato_01.png"),
    os.path.join(base, "Perros", "perro_01.png"),
    os.path.join(base, "Caballos", "caballo_01.png"),
    os.path.join(base, "Aves", "ave_01.jpg"),
    os.path.join(base, "Pinguinos", "pinguino_01.png"),
    os.path.join(base, "Hipopotamos", "hipopotamo_01.jpg"),
]

for ruta in imagenes_prueba:
    try:
        predecir_y_comparar(ruta, model_custom, model_tl, class_names, IMG_HEIGHT, IMG_WIDTH)
    except Exception as e:
        print("No se pudo cargar", ruta, "->", e)

print("\nListo, los .csv quedan para graficar en el informe")
