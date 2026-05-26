import tkinter as tk
from tkinter import filedialog

from PIL import Image, ImageTk

import tensorflow as tf
import numpy as np

from tensorflow.keras.utils import (
    load_img,
    img_to_array
)



# ЗАГРУЗКА МОДЕЛИ
model = tf.keras.models.load_model(
    "ship_cnn_model.keras"
)



# КЛАССЫ
class_names = [
    "ContainerShip",
    "Destroyer",
    "FishingBoat",
    "OilTanker",
    "PassengerShip",
    "Submarine"
]



# ФУНКЦИЯ ПРЕДСКАЗАНИЯ
def predict_image():

    file_path = filedialog.askopenfilename(
        filetypes=[
            (
                "Image Files",
                "*.jpg *.png *.jpeg"
            )
        ]
    )

    if not file_path:
        return


    # ОТОБРАЖЕНИЕ КАРТИНКИ
    image = Image.open(file_path)

    image = image.resize((300, 300))

    photo = ImageTk.PhotoImage(image)

    image_label.config(image=photo)

    image_label.image = photo



    # ПОДГОТОВКА ДЛЯ CNN
    img = load_img(
        file_path,
        target_size=(128, 128)
    )

    img_array = img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )



    # ПРЕДСКАЗАНИЕ
    predictions = model.predict(
        img_array,
        verbose=0
    )

    predicted_class = class_names[
        np.argmax(predictions)
    ]

    confidence = np.max(predictions)



    # ВЫВОД РЕЗУЛЬТАТА
    result_label.config(
        text=
        f"Prediction: {predicted_class}\n"
        f"Confidence: {confidence:.2f}"
    )



# GUI
root = tk.Tk()

root.title(
    "Ship Classification CNN"
)

root.geometry("600x600")



# КНОПКА
button = tk.Button(
    root,
    text="Choose Image",
    command=predict_image,
    font=("Arial", 14)
)

button.pack(pady=20)



# КАРТИНКА
image_label = tk.Label(root)

image_label.pack()



# РЕЗУЛЬТАТ
result_label = tk.Label(
    root,
    text="Result will appear here",
    font=("Arial", 14)
)

result_label.pack(pady=20)



# ЗАПУСК
root.mainloop()