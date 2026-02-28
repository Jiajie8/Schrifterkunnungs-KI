import numpy as np
import tensorflow as tf
from keras import layers, models


# Daten laden
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test  = np.load("X_test.npy")
y_test  = np.load("y_test.npy")

X_train = X_train/255
X_test = X_test/255

model = models.Sequential([
    
    # Input: 32x32 grayscale image
    layers.Input(shape=(32, 32, 1)),
    
    # 1. Convolution Block
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),   # 32x32 -> 16x16
    
    # 2. Convolution Block
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),   # 16x16 -> 8x8
    
    # 3. Convolution Block
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),   # 8x8 -> 4x4
    
    # Flatten
    layers.Flatten(),
    
    # Fully Connected Layer
    layers.Dense(128, activation='relu'),
    
    # Output Layer (26 Klassen)
    layers.Dense(26, activation='softmax')
])

# Kompilieren
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=15, batch_size=64)
model.evaluate(X_test,y_test)

pred = model.predict(X_test)

# Index → Buchstabe
def index_to_letter(i): 
    return chr(i + 65)

correct = 0
# Ausgabe für alle Testbilder
for i in range(len(X_test)):
    predicted_letter = index_to_letter(np.argmax(pred[i]))
    true_letter = index_to_letter(y_test[i][0])

    print(f"Vorhersage: {predicted_letter}  | Wahr: {true_letter}")

    if predicted_letter == true_letter:
        correct += 1

acc = correct / len(X_test) * 100

print(f"Manuelle Genauigkeit:{acc}%")

# Modellübersicht anzeigen
model.summary() 

model.save("buchstaben_model_cnn.keras")