import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import EarlyStopping #type: ignore
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import csv



dataset = "DATASET_CSV/"
train_folder = dataset + "Train/"
test_folder = dataset + "Test/"
val_folder = dataset + "Validation/"

### PARAMETRES

nb_img = 1800
nb_test = 600
nb_val = 600

#list_char = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'Nothing', 'O', 'P', 'Q', 'R', 'S', 'Space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
list_char = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'Space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

### FIN PARAMETRES

def csv_to_array(input_path):

    X = []
    with open(input_path, mode='r', newline='', encoding='utf-8') as infile :
        
        reader = csv.reader(infile)
        for row in reader:
            new_row = []
            for array in row:
                new_array = array.replace('[', '').replace(']','').replace(',', '').split(' ')
                new_array = [float(i) for i in new_array]
                new_row.append(new_array)
            X.append(new_row)
    return np.array(X)




def load_set_label(dir, lim, label):
    
    X = csv_to_array(dir + label + ".csv")[:lim]
    Y = np.array([list_char.index(label)]*lim)

    return X,Y


def load_set(dir, lim, list_labels):
    X, Y = load_set_label(dir, lim, list_labels[0])
    for label in list_labels[1:]:
        Nx, Ny = load_set_label(dir, lim, label)
        X = np.concatenate((X, Nx), axis=0)
        Y = np.concatenate((Y, Ny), axis=0)

    return np.array(X), np.array(Y)

X_train, Y_train = load_set(train_folder, nb_img, list_char)
X_test, Y_test = load_set(test_folder, nb_test, list_char)
X_val, Y_val = load_set(val_folder, nb_val, list_char)



X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)


X_train_flat = X_train.reshape(X_train.shape[0], -1)  # (n_samples, 63)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)


print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
print(X_val.shape, Y_val.shape)

X_train, Y_train = shuffle(X_train, Y_train, random_state=42)

print(f"size training set {len(X_train)}")
print(f"size test set {len(X_test)}")
print(f"dimension {len(X_train[0])} x {len(X_train[0][0])}")


model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(21,3)),
    BatchNormalization(),
    Dropout(0.3),
    
    Conv1D(128, kernel_size=3, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(27, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=3,           # stop si aucune amélioration pendant 3 epochs
    restore_best_weights=True
)

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=64, epochs = 30, validation_data = (X_val, Y_val), callbacks=[early_stop])

print("\nEvaluation")
model.evaluate(X_test, Y_test)


model.save("mon_modele_ASL.keras")