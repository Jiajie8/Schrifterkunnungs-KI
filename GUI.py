import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =========================
# Modell laden
# =========================
model = tf.keras.models.load_model("buchstaben_model_cnn.keras")

# =========================
# Fenster
# =========================
root = tk.Tk()
root.title("CNN Buchstaben Live-Prediction")

canvas_size = 200
canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg="white")
canvas.grid(row=0, column=0, padx=10, pady=10)

# PIL Bild für Zeichnung
image = Image.new("L", (canvas_size, canvas_size), 255)
draw = ImageDraw.Draw(image)

# =========================
# Matplotlib Setup
# =========================
fig, ax = plt.subplots(figsize=(8, 3))
letters = [chr(i + 65) for i in range(26)]
bars = ax.bar(letters, [0]*26)
ax.set_ylim(0, 1)
ax.set_title("Wahrscheinlichkeiten")

canvas_plot = FigureCanvasTkAgg(fig, master=root)
canvas_plot.get_tk_widget().grid(row=1, column=0)

result_label = tk.Label(root, text="Vorhersage: -", font=("Arial", 22))
result_label.grid(row=2, column=0, pady=10)

# =========================
# Zeichnen
# =========================
def paint(event):
    x1, y1 = event.x-15, event.y-15
    x2, y2 = event.x+15, event.y+15
    canvas.create_oval(x1, y1, x2, y2, fill="black")
    draw.ellipse([x1, y1, x2, y2], fill="black")

canvas.bind("<B1-Motion>", paint)

# =========================
# Bild für CNN vorbereiten
# =========================
def preprocess_gui_image(img, target_size=32):
    arr = np.array(img)
    coords = np.argwhere(arr < 255)  # schwarze Pixel
    if coords.size == 0:
        return None  # leer
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    cropped = arr[y0:y1+1, x0:x1+1]
    pil_crop = Image.fromarray(cropped)
    pil_crop = pil_crop.resize((target_size, target_size), Image.Resampling.LANCZOS)
    arr = np.array(pil_crop)/255.0
    arr = arr.reshape(1, target_size, target_size, 1)
    return arr

# =========================
# Live Prediction
# =========================
after_id = None

def live_predict():
    global after_id
    img_input = preprocess_gui_image(image)
    if img_input is not None:
        pred = model.predict(img_input, verbose=0)[0]
        index = np.argmax(pred)
        letter = chr(index + 65)
        result_label.config(text=f"Vorhersage: {letter}")
        for bar, p in zip(bars, pred):
            bar.set_height(p)
        canvas_plot.draw_idle()
    else:
        result_label.config(text="Vorhersage: -")
        for bar in bars:
            bar.set_height(0)
        canvas_plot.draw_idle()
    after_id = root.after(300, live_predict)

live_predict()

# =========================
# Löschen
# =========================
def clear():
    canvas.delete("all")
    global image, draw
    image = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(image)
    result_label.config(text="Vorhersage: -")
    for bar in bars:
        bar.set_height(0)
    canvas_plot.draw_idle()

tk.Button(root, text="Löschen", command=clear).grid(row=3, column=0, pady=10)

# =========================
# Sauberes Schließen
# =========================
def on_closing():
    global after_id
    if after_id is not None:
        root.after_cancel(after_id)
    root.destroy()
    plt.close(fig)

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()