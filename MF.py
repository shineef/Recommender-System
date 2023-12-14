import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dot
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def train_model(ratings, num_users, num_books):
    # Split the data into a training set and a validation set
    X = ratings[['User-ID', 'ISBN']].values
    y = ratings['Book-Rating'].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = pd.to_numeric(y_train, errors='coerce')
    y_val = pd.to_numeric(y_val, errors='coerce')

    X_train = X_train[~np.isnan(y_train)]
    y_train = y_train[~np.isnan(y_train)]
    X_val = X_val[~np.isnan(y_val)]
    y_val = y_val[~np.isnan(y_val)]

    # Define the model
    user_input = Input(shape=(1,))
    user_embedding = Embedding(num_users, 10)(user_input)
    user_flatten = Flatten()(user_embedding)

    book_input = Input(shape=(1,))
    book_embedding = Embedding(num_books, 10)(book_input)
    book_flatten = Flatten()(book_embedding)

    dot_product = Dot(axes=1)([user_flatten, book_flatten])

    model = Model(inputs=[user_input, book_input], outputs=dot_product)

    # Compile the model
    # model.compile(loss='mean_squared_error', optimizer=Adam())
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=[tf.keras.metrics.MeanSquaredError()])

    # print(y_train)
    # print(y_val)
    # Train the model
    history = model.fit([X_train[:, 0], X_train[:, 1]], y_train, batch_size=64, epochs=5, verbose=1, validation_data=([X_val[:, 0], X_val[:, 1]], y_val))

    # Save the model
    model.save('book_recommender_model.h5')
    _, mse = model.evaluate([X_val[:, 0], X_val[:, 1]], y_val, verbose=0)
    print('Validation MSE: ', mse)

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.tight_layout()
    plt.show()

    return model
