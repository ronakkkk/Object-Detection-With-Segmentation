import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas
from cnn_model import cnn_model
from mnist_format_images import imageprepare
from svm_classifier import svm_model

if __name__ == '__main__':
    # calling model
    cnn = cnn_model()
    svm_mod = svm_model()
    # /*Load iamge in grayscale mode*/
    image = cv2.imread("sample3 (1).jpg")

    # Apply Gaussian blur
    image = cv2.GaussianBlur(image, (5, 5), 2)

    # Apply median blur
    image = cv2.medianBlur(image, 3)

    out = np.zeros(image.shape,np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgplot = plt.imshow(gray)
    plt.show()

    # using threshold to extract digits
    th, image_threshold = cv2.threshold(gray, 128, 192, cv2.THRESH_OTSU)

    print(th)
    # 202

    imgplot = plt.imshow(image_threshold)
    plt.show()


    blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

    #################      Now finding image_contours         ###################

    image_contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    samples =  np.empty((0,100))
    responses = []
    keys = [i for i in range(48,58)]
    c=0
    for cnt in image_contours:
        if cv2.contourArea(cnt)>50:
            [x,y,w,h] = cv2.boundingRect(cnt)
            if h>28:
                img = cv2.rectangle(image_threshold,(x,y),(x+w,y+h),(0,0,255),2)
                roi = thresh[y:y+h,x:x+w]
                roismall = cv2.resize(roi, (10, 10))
                cv2.imshow('norm', img)
                key = cv2.waitKey(0)
                digit = img[y:y + h, x:x + w]
                digit = (255 - digit)
                cv2.imshow('norm',digit)
                key = cv2.waitKey(0)

                # saving the image
                c=c+1
                filename = str(c)+".jpg"
                cv2.imwrite(filename,digit)

                # resize the digit image
                # resizing the image for processing them in the covnet
                img = imageprepare(filename)

                '''CNN'''
                # final step-forming the data with numpy array of the digit
                data_img_lst = np.array(img)
                data_img_digit = data_img_lst.reshape(-1, 28, 28, 1)
                res = cnn.predict(data_img_digit)
                print("Predicting image digit using CNN classifier")
                # print(res)
                print(np.argmax(res))

                '''SVM'''
                print("Predicting image digit using SVM classifier")
                df = pandas.DataFrame()
                temp_df = pandas.Series(img)
                df = df.append(temp_df, ignore_index=True)
                # print(df.head())
                print(svm_mod.predict(df))




