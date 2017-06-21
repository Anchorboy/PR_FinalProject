import os
import random
from itertools import cycle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from skimage import exposure
from skimage.feature import hog
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC


def read_data():
    img_size = (200, 200)

    train_all = []
    test_all = []
    current_base = os.path.abspath('.')
    train_path = os.path.join(current_base, "train")
    test_path = os.path.join(current_base, "test")

    # read train
    for dir_name in os.listdir(train_path):
        dir_path = os.path.join(train_path, dir_name)
        class_id = int(dir_name)

        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            img_vec = cv2.imread(img_path, flags=0)
            # print img_vec.shape
            # res = cv2.resize(img_vec, (int(img_vec.shape[0]*0.5), int(img_vec.shape[1]*0.5)), interpolation=cv2.INTER_CUBIC)
            res = cv2.resize(img_vec, img_size, interpolation=cv2.INTER_CUBIC)
            nor_res = np.zeros_like(res)
            nor_res = cv2.normalize(src=res, dst=nor_res, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            train_all.append((class_id, nor_res))

    # read test
    for dir_name in os.listdir(test_path):
        dir_path = os.path.join(test_path, dir_name)
        class_id = int(dir_name)

        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            img_vec = cv2.imread(img_path, flags=0)
            # print img_vec.shape
            # res = cv2.resize(img_vec, (int(img_vec.shape[0]*0.5), int(img_vec.shape[1]*0.5)), interpolation=cv2.INTER_CUBIC)
            res = cv2.resize(img_vec, img_size, interpolation=cv2.INTER_CUBIC)
            nor_res = np.zeros_like(res)
            nor_res = cv2.normalize(src=res, dst=nor_res, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            test_all.append((class_id, nor_res))

    return train_all, test_all

def show_hog(img):
    fd, hog_img = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), feature_vector=True,
                      visualise=True)
    h = hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(3, 3), feature_vector=True,
            visualise=False)
    print fd.shape
    print hog_img.shape
    print h.shape
    print hog_img
    hog_image_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 0.02))
    print hog_image_rescaled
    cv2.imshow("ori", img)
    cv2.imshow("hog", hog_image_rescaled)
    cv2.waitKey()

def draw_multi_ROC(y_test, decf, score):
    y_test_bin = label_binarize(y_test, classes=[(i+1) for i in xrange(n_classes)])

    # Compute macro-average ROC curve and ROC area
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test_bin[:, i], decf[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test_bin.ravel(), decf.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    # print mean_tpr
    for i in range(n_classes):
        # print interp(all_fpr, fpr[i], tpr[i])
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(20,10))
    plt.title("Score: " + str(score))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    lw = 2

    colors = cycle([ ((round(random.uniform(0.0, 1.0), 2)), (round(random.uniform(0.0, 1.0), 2)), (round(random.uniform(0.0, 1.0), 2)))
                     for i in xrange(n_classes) ])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc='lower right')
    if not os.path.exists("task2"):
        os.mkdir("task2")
    plt.savefig("task2/"+str(pix_cell)+'.png')
    plt.show()
    # plt.close()

if __name__ == "__main__":
    train_all, test_all = read_data()
    X_train = [ i[1] for i in train_all ]
    y_train = [ i[0] for i in train_all ]

    X_test = [ i[1] for i in test_all ]
    y_test = [ i[0] for i in test_all ]

    n_classes = len(set(y_train))
    pix_cell = 8
    while pix_cell < 16.1:
        print pix_cell
        X_hog_train = [hog(train_img, orientations=9, pixels_per_cell=(pix_cell, pix_cell), cells_per_block=(1, 1), feature_vector=True, visualise=False)
                       for train_img in X_train]
        X_hog_test = [hog(test_img, orientations=9, pixels_per_cell=(pix_cell, pix_cell), cells_per_block=(1, 1), feature_vector=True,visualise=False)
                        for test_img in X_test]

        svm = SVC()
        svm.fit(X_hog_train, y_train)

        predict = svm.predict(X_hog_test)
        score = svm.score(X_hog_test, y_test)
        svm.decision_function_shape = "ovr"
        decf = svm.decision_function(X_hog_test)

        draw_multi_ROC(y_test, decf, score)
        pix_cell += 1