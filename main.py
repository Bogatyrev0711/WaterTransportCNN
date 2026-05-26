import tensorflow as tf
import matplotlib.pyplot as plt



# 1. ДАТАСЕТ
dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=(128, 128),
    batch_size=32,
    shuffle=True
)

class_names = dataset.class_names
print("Классы:", class_names)



# 2. НОРМАЛИЗАЦИЯ
normalization_layer = tf.keras.layers.Rescaling(1. / 255)

dataset = dataset.map(
    lambda x, y: (normalization_layer(x), y)
)



# 3. SPLIT
dataset_size = len(dataset)

train_size = int(0.8 * dataset_size)

train_ds = dataset.take(train_size)
val_ds = dataset.skip(train_size)



# 4. ВИЗУАЛИЗАЦИЯ
plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):

    for i in range(9):

        plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy())

        plt.title(class_names[labels[i]])

        plt.axis("off")

plt.show()



# 5. CNN МОДЕЛЬ
model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(128, 128, 3)),

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation='relu'
    ),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation='relu'
    ),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation='relu'
    ),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(
        len(class_names),
        activation='softmax'
    )
])


# 6. COMPILE
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)



# 7. TRAIN
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)



# 8. ГРАФИКИ

# ACCURACY
plt.figure(figsize=(8, 5))

plt.plot(
    history.history['accuracy'],
    label='Train Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title("Model Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.savefig("accuracy_graph.png")

plt.show()


# LOSS
plt.figure(figsize=(8, 5))

plt.plot(
    history.history['loss'],
    label='Train Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title("Model Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.savefig("loss_graph.png")

plt.show()



# 9. СОХРАНЕНИЕ
model.save("ship_cnn_model.keras")

print("Модель сохранена!")
print("Графики сохранены!")