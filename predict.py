import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.utils import load_img, img_to_array



# 1. ЗАГРУЗКА МОДЕЛИ
model = tf.keras.models.load_model(
    "ship_cnn_model.keras"
)



# 2. КЛАССЫ
class_names = [
    "ContainerShip",
    "Destroyer",
    "FishingBoat",
    "OilTanker",
    "PassengerShip",
    "Submarine"
]



# 3. ПАПКА С ТЕСТАМИ
test_folder = "test_images"



# 4. ПРОВЕРКА ВСЕХ КАРТИНОК
for filename in os.listdir(test_folder):

    if filename.endswith((".jpg", ".png", ".jpeg")):

        img_path = os.path.join(
            test_folder,
            filename
        )

        # загрузка изображения
        img = load_img(
            img_path,
            target_size=(128, 128)
        )

        # преобразование
        img_array = img_to_array(img)

        img_array = img_array / 255.0

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # prediction
        predictions = model.predict(
            img_array,
            verbose=0
        )

        predicted_class = class_names[
            np.argmax(predictions)
        ]

        confidence = np.max(predictions)

        # вывод результата
        print("=" * 40)

        print("Файл:", filename)

        print(
            "Предсказание:",
            predicted_class
        )

        print(
            f"Уверенность: {confidence:.2f}"
        )