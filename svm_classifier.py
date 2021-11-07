import pandas
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from mnist_format_images import imageprepare
from sklearn.svm import SVC
import seaborn as sns
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


def remove_constant_pixels(pixels_df):
    """Removes from the images the pixels that have a constant intensity value,
    either always black (0) or white (255)
    Returns the cleared dataset & the list of the removed pixels (columns)"""

    # Remove the pixels that are always black to compute faster
    changing_pixels_df = pixels_df.loc[:]
    dropped_pixels_b = []

    # Pixels with max value =0 are pixels that never change
    for col in pixels_df:
        if changing_pixels_df[col].max() == 0:
            changing_pixels_df.drop(columns=[col], inplace=True)
            dropped_pixels_b.append(col)
    print("Constantly black pixels that have been dropped: {}".format(dropped_pixels_b))

    # Same with pixels with min=255 (white pixels)
    dropped_pixels_w = []
    for col in changing_pixels_df:
        if changing_pixels_df[col].min() == 255:
            changing_pixels_df.drop(columns=[col], inplace=True)
            dropped_pixels_w.append(col)
    print("\n Constantly white pixels that have been dropped: {}".format(dropped_pixels_b))

    print(changing_pixels_df.head())
    print("Remaining pixels: {}".format(len(changing_pixels_df.columns)))
    print("Pixels removed: {}".format(784 - len(changing_pixels_df.columns)))

    return changing_pixels_df, dropped_pixels_b + dropped_pixels_w

def convert_img(handwritten_data, name):
    # tqdm is only used for interactive loading
    # loading the training data
    df = pandas.DataFrame()
    for img in tqdm(os.listdir(handwritten_data)):
        path = os.path.join(handwritten_data, img)

        img = imageprepare(path)
        temp_df = pandas.Series(img)
        df = df.append(temp_df, ignore_index=True)
        df['label'] = int(name)


    return df

def svm_model():
    '''Pre-Processing'''
    # Convert images into mnist data format
    df_0 = convert_img(digit_0, "0")
    df_1 = convert_img(digit_1, "1")
    df_2 = convert_img(digit_2, "2")
    df_3 = convert_img(digit_3, "3")
    df_4 = convert_img(digit_4, "4")
    df_5 = convert_img(digit_5, "5")
    df_6 = convert_img(digit_6, "6")
    df_7 = convert_img(digit_7, "7")
    df_8 = convert_img(digit_8, "8")
    df_9 = convert_img(digit_9, "9")


    # appending dataframe
    df = df_0.append(df_1)
    df = df.append(df_2)
    df = df.append(df_3)
    df = df.append(df_4)
    df = df.append(df_5)
    df = df.append(df_6)
    df = df.append(df_7)
    df = df.append(df_8)
    df = df.append(df_9)
    df = df.sample(frac=1).reset_index(drop=True)
    print(df)
    # df_pixels, df_dropped = remove_constant_pixels(df.drop(columns='label'))
    # df_pixels['labels'] = df['labels']

    # split the dataset
    X = df.drop(columns='label')
    train_x, test_x, train_y, test_y = train_test_split(X, df['label'])


    # using dataframe try to find any null values
    print("Printing any nan values in the dataset: ", X.isnull().sum())

    # plot data
    _ = df['label'].value_counts().plot(kind='bar')
    plt.show()


    svm_class = SVC()

    svm_class.fit(train_x, train_y)
    svm_pred = svm_class.predict(test_x)

    # evaluation
    conf_matrix = confusion_matrix(test_y, svm_pred)
    print("report", classification_report(test_y, svm_pred))
    print("Accuracy:", accuracy_score(test_y, svm_pred))
    acc_score = accuracy_score(test_y, svm_pred)
    plt.figure(figsize=(9, 9))
    sns.heatmap(conf_matrix, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(acc_score)
    plt.title(all_sample_title, size=15)
    plt.show()
    return svm_class
