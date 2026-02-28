import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input


# Daten laden
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test  = np.load("X_test.npy")
y_test  = np.load("y_test.npy")

X_train = X_train/255
X_test = X_test/255


#Modell
model = Sequential([
    Input(shape=(32, 32)),     # images are 32x32
    Flatten(),
    Dense(64, activation='relu'),
    Dense(128
          , activation='relu'),
    Dense(26, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=1000, batch_size=64)
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

model.save("buchstaben_model.keras")