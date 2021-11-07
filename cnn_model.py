import random
import tensorflow
import pandas
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from mnist_format_images import imageprepare
'''read file'''
digit_0 = 'trainingSet/0'
digit_1 = 'trainingSet/1'
digit_2 = 'trainingSet/2'
digit_3 = 'trainingSet/3'
digit_4 = 'trainingSet/4'
digit_5 = 'trainingSet/5'
digit_6 = 'trainingSet/6'
digit_7 = 'trainingSet/7'
digit_8 = 'trainingSet/8'
digit_9 = 'trainingSet/9'

SIZE_IMG = 28
learning_rate = 0.01

'''Setting up the model which will help with tensorflow models'''
MODEL_NAME = 'digits-{}-{}.model'.format(learning_rate, '6conv-basic')

def convert_img(handwritten_data, name):
    data_img_lst = []
    # tqdm is only used for interactive loading
    # loading the training data
    df = pandas.DataFrame()
    for img in tqdm(os.listdir(handwritten_data)):
        path = os.path.join(handwritten_data, img)

        img = imageprepare(path)
        # final step-forming the training data list with numpy array of the images
        data_img_lst.append([img, numpy.array(int(name))])

    return data_img_lst

def cnn_model():
    '''Pre-Processing'''
    # Convert images into mnist data format
    data_0 = convert_img(digit_0, "0")
    data_1 = convert_img(digit_1, "1")
    data_2 = convert_img(digit_2, "2")
    data_3 = convert_img(digit_3, "3")
    data_4 = convert_img(digit_4, "4")
    data_5 = convert_img(digit_5, "5")
    data_6 = convert_img(digit_6, "6")
    data_7 = convert_img(digit_7, "7")
    data_8 = convert_img(digit_8, "8")
    data_9 = convert_img(digit_9, "9")

    # appending data
    data = data_0
    data.extend(data_1)
    data.extend(data_2)
    data.extend(data_3)
    data.extend(data_4)
    data.extend(data_5)
    data.extend(data_6)
    data.extend(data_7)
    data.extend(data_8)
    data.extend(data_9)

    # shuffle the data
    random.shuffle(data)

    # convert to dataframe
    data_df = pandas.DataFrame(data)

    # using dataframe try to find any null values
    print("Printing any nan values in the dataset: ", data_df.isnull().sum())

    # plot data
    _ = data_df[1].value_counts().plot(kind='bar')
    plt.show()
    # split data
    train = data[:-41000]
    test = data[-41000:]

    # Reshape data
    # X-Features & Y-Labels
    X = numpy.array([i[0] for i in train]).reshape(-1, 28, 28, 1)
    Y = numpy.array([i[1] for i in train])
    test_x = numpy.array([i[0] for i in test]).reshape(-1, 28, 28, 1)
    test_y = numpy.array([i[1] for i in test])
    input_shape = (28, 28, 1)

    '''Deep Learning Model'''
    cnn_dlm = tensorflow.keras.models.Sequential()
    cnn_dlm.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

    cnn_dlm.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape))
    BatchNormalization()
    cnn_dlm.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_dlm.add(Dropout(0.2))

    cnn_dlm.add(Flatten())

    cnn_dlm.add(Dense(128, activation='relu'))
    BatchNormalization()
    cnn_dlm.add(Dropout(0.2))

    cnn_dlm.add(Dense(10, activation='sigmoid'))
    # compile
    cnn_dlm.compile(
        optimizer='rmsprop',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print(cnn_dlm.summary())
    cnn_dlm.fit(x=X, y=Y, epochs=20)

    print(cnn_dlm.evaluate(test_x, test_y))
    return cnn_dlm
