import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import tensorflow as tf
layers, models, optimizers, ImageDataGenerator, regularizers = tf.keras.layers, tf.keras.models, tf.keras.optimizers, tf.keras.preprocessing.image.ImageDataGenerator, tf.keras.regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from random import sample, randint
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import keract
import os
import json

def show(img):
    cv2.imshow('img', img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return(key)

def preProcessing(img, image_size):
    # r_img = img[60:135, :, :]
    r_img = img[40:135, :, :]
    # TODO convert BGR2YUV
    # r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2YUV)
    r_img = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
    # r_img = cv2.GaussianBlur(r_img, (3, 3), 0)
    r_img = cv2.resize(r_img, image_size)
    r_img = r_img / 255

    return r_img

def remove_randomely(df, portion=0.1):
    idxs = df[df['steering'] == 0].index
    should_remove_idxs = sample(tuple(idxs), k=int(len(idxs)*(1-portion)))
    rdf = df.drop(index=should_remove_idxs)
    return rdf

def load_data(image_size):
    df = pd.read_csv(r"C:\My Files\Datasets\car simulator-windows-64\driving_log.csv")
    # plt.figure(dpi=300)
    # sns.histplot(df['steering'], color='tab:blue')
    # plt.show()
    df = remove_randomely(df, portion=0.05)
    plt.figure(dpi=300)
    sns.histplot(df['steering'], color='tab:red')
    plt.savefig('histplot.jpg')
    # plt.show()

    dataset = []
    labels = []
    addrs = []
    is_flips = []
    count = df.shape[0]

    for i in df.index:
        path = df.loc[i, 'center']
        steer = df.loc[i, 'steering']

        img = cv2.imread(path)
        r_img = preProcessing(img, image_size)

        for is_flip in [False, True]:
            if is_flip:
                r_img = cv2.flip(r_img, 1)
                steer = -1*steer

            dataset.append(r_img)
            labels.append(steer)
            addrs.append(path)
            is_flips.append(is_flip)

        if i%500 == 0:
            print(f'{i} / {count} loaded')

    dataset = np.array(dataset, dtype='float32')
    labels = np.array(labels, dtype='float32')
    is_flips = np.array(is_flips)

    # labels = labels/180 # normalizing Angle -1 to 1
    # Todo for balancing data
    rus = RandomUnderSampler()
    ros = RandomOverSampler()

    x_train, x_test, y_train, y_test, addrs_train, addrs_test, is_flip_train, is_flip_test = train_test_split(dataset, labels, addrs, is_flips,
                                                        random_state=3,test_size=0.2)

    return x_train, x_test, y_train, y_test, addrs_train, addrs_test, is_flip_train, is_flip_test


def add_guassian_noise(img):
    noise = np.random.normal(0, 0.01, img.shape)
    return img+noise

def read_checkpoint():
    with open('checkpoint.txt') as f:
        checkpoint = f.read()
    return int(checkpoint)
def define_model(image_size, optimizer):
    input_shape = list(reversed(image_size)) + [3]  # becausue (32, 32, 3)

    # Suggested by PDF
    # model = models.Sequential([
    #
    #     # layers.BatchNormalization(input_shape=input_shape),
    #     layers.Conv2D(24, (5, 5), strides=(2, 2), padding='valid', input_shape=input_shape),
    #     layers.Conv2D(36, (5, 5), strides=(2, 2), padding='valid'),
    #     layers.Conv2D(48, (5, 5), strides=(2, 2), padding='valid'),
    #     layers.Conv2D(64, (3, 3), strides=(1, 1), padding='valid'),
    #     layers.Conv2D(64, (3, 3), strides=(1, 1), padding='valid'),
    #     # layers.BatchNormalization(),
    #     layers.Flatten(),
    #
    #     # layers.Dense(1164, activation='relu'),
    #     layers.Dense(300, activation='relu'),
    #     layers.Dropout(0.4),
    #     layers.Dense(50, activation='relu'),
    #     layers.Dropout(0.4),
    #     layers.Dense(10, activation='relu'),
    #     layers.Dense(1, activation='linear')
    # ])

    # -------------Transfer learning------------------------



    # baseModel = tf.keras.applications.ResNet50(weights='imagenet',
    #                   include_top=False,
    #                   input_tensor=layers.Input(shape=input_shape))

    baseModel = tf.keras.applications.VGG16(weights='imagenet',
                      include_top=False,
                      input_tensor=layers.Input(shape=input_shape))
    model = models.Sequential([
        baseModel,
        layers.Flatten(),
        # layers.Dense(700, activation='relu',
        #              # kernel_regularizer=regularizers.l2(0.01)
        #              ),
        # layers.BatchNormalization(),
        # layers.Dropout(0.4),
        layers.Dense(124, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        # layers.Dense(40, activation='relu'),
        # layers.BatchNormalization(),
        # layers.Dropout(0.4),

        layers.Dense(1, activation='linear')
    ])

    for layer in baseModel.layers:
        layer.trainable = False

    # -----------developed by myself
    # model = models.Sequential([
    #     layers.Conv2D(32, (4, 4), strides=(2, 2), padding='valid', input_shape=input_shape),
    #     layers.LeakyReLU(alpha=0.2),
    #     # layers.Dropout(0.3),
    #
    #     layers.Conv2D(64, (4, 4), strides=(2, 2), padding='valid'),
    #     layers.LeakyReLU(0.2),
    #     # layers.Dropout(0.3),
    #
    #     layers.Conv2D(128, (4, 4), strides=(2, 2), padding='valid'),
    #     layers.LeakyReLU(0.2),
    #     # layers.Dropout(0.3),
    #
    #     layers.Conv2D(128, (4, 4), strides=(2, 2), padding='valid'),
    #     layers.LeakyReLU(0.2),
    #     # layers.Dropout(0.3),
    #
    #     layers.Flatten(),
    #     layers.Dense(128, activation='relu'),
    #     layers.BatchNormalization(),
    #     layers.LeakyReLU(0.2),
    #     layers.Dropout(0.3),
    #
    #     layers.Dense(10, activation='relu'),
    #     layers.BatchNormalization(),
    #     layers.LeakyReLU(0.2),
    #     layers.Dropout(0.3),
    #
    #     layers.Dense(1, activation='linear')
    # ])

    model.compile(optimizer=optimizer,
                  loss='mse',
                  # metrics=['mape']
                  )

    model.summary()
    return model

def training(model, x_train, y_train, x_val, y_val,
             epochs, batch_size, optimizer, image_size, start_from_epoch):


    aug = ImageDataGenerator(
        rotation_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest',
        # brightness_range=[0.8, 1],
        # cval=0,
        preprocessing_function=add_guassian_noise
    )

    steps_per_epoch = len(x_train)//batch_size

    checkpoint = ModelCheckpoint(
        'model.h5',  # Filepath to save the best model
        monitor='val_loss',  # Metric to monitor (e.g., validation loss)
        # monitor='val_accuracy',
        save_best_only=True,  # Save only the best model (overwrite previous best)
        mode='min',  # 'min' if monitoring loss, 'max' if monitoring accuracy
        # mode='max',
        verbose=1  # Display progress
    )

    H = model.fit(aug.flow(x_train, y_train,
                         batch_size=batch_size, seed=3),
                steps_per_epoch=steps_per_epoch,
                epochs=epochs, batch_size=batch_size,
                validation_data=(x_val, y_val),
                callbacks=[checkpoint]
                )

    with open('checkpoint.txt', 'w') as f:
        f.write(str(epochs+start_from_epoch))

    return H.history, model

def write_history(history, start_from_epoch):

    if start_from_epoch != 1:
        with open('result.json') as f:
            past = json.load(f)

        for metric, values in history.items():
            past[metric].extend(values)
        history = past

    with open('result.json', 'w') as f:
        json.dump(history, f)

    return history



def show_result(model, history, x_test, y_test, epochs):
    y_pred = model.predict(x_test)

    # loss, acc = model.evaluate(x_test, y_test)

    loss = model.evaluate(x_test, y_test)
    print(f'{loss=}')

    r2 = r2_score(y_test, y_pred)
    print(f'R2 Score: {r2}')
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f'MAPE : {mape}')

    sns.set_theme('paper', 'white', 'viridis')
    plt.figure(dpi=300)
    for metric, value in history.items():
        plt.plot(value, label=metric)
        # break # because I just want to plot loss

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('metrics')
    plt.savefig('training_plot.jpg')
    plt.show()


def check_with_images(model, addrs_test,is_flip_test, image_size, how_many_times):
    for i in range(how_many_times):
        rnd = randint(0, len(addrs_test) - 1)

        img = cv2.imread(addrs_test[rnd])
        r_img = preProcessing(img, image_size)

        is_flip = is_flip_test[rnd]
        # print(f'{is_flip=}')

        y_true = y_test[rnd]
        # print(f'{y_true=}')
        if is_flip:
            y_true = y_true * -1
            # print(f'{y_true=}')

        y_pred = model.predict(np.array([r_img]))[0][0]
        # print('f{y_pred=}')

        cv2.rectangle(img, (0, 0), (120, 30), (255, 255, 255), -1)

        cv2.putText(img, 'Pred :' + str(y_pred), (3, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        cv2.putText(img, 'True :' + str(y_true), (3, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        cv2.imshow('image', img)
        # cv2.imshow('img', r_img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # for the last iteme
    # activations = keract.get_activations(model, np.array([r_img]))
    # keract.display_activations(activations, save=False)

#
def main(epochs=5, learning_rate=1e-3):
    image_size = [200, 66]
    batch_size = 32
    # epochs = 15
    # optimizer = optimizers.Adam(learning_rate=1e-3, decay=2.5e-5)
    optimizer = optimizers.Adam(learning_rate=learning_rate, decay=2.5e-5)

    if not os.path.exists(r"checkpoint.txt"):
        model = define_model(image_size, optimizer)
        start_from_epoch = 1
    else:
        model = models.load_model("model.h5")
        start_from_epoch = read_checkpoint()
        print(f'\n\n\n\n\n>>Started from {start_from_epoch}>>>')

    history, model = training(model, x_train, y_train, x_test, y_test,
                        epochs, batch_size, optimizer, image_size, start_from_epoch)

    history = write_history(history, start_from_epoch)
    show_result(model, history, x_test, y_test, epochs)
    # check_with_images(model, addrs_test, is_flip_test, image_size, how_many_times=5)

# %%
image_size = [200, 66]
x_train, x_test, y_train, y_test, addrs_train, addrs_test, is_flip_train, is_flip_test = load_data(image_size)
main(epochs=4, learning_rate=1e-3)
main(epochs=3, learning_rate=1e-4)
