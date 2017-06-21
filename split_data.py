import os
import cv2
import random
import shutil
import numpy as np

def split_img(input_path):
    split_ratio = 0.8

    for dir_name in xrange(10):
        dir_name += 1
        dir_name = str(dir_name)

        dir_path = os.path.join(input_path, dir_name)
        img_in_class = os.listdir(dir_path)
        rand_train_img = set(random.sample(img_in_class, int(len(img_in_class) * split_ratio)))
        rand_test_img = set(img_in_class) - rand_train_img

        for img_name in rand_train_img:
            img_path = os.path.join(dir_path, img_name)
            if not os.path.exists("train/"+dir_name):
                os.mkdir("train/"+dir_name)
            shutil.copyfile(img_path, "train/"+dir_name+"/"+img_name)

        for img_name in rand_test_img:
            img_path = os.path.join(dir_path, img_name)
            if not os.path.exists("test/"+dir_name):
                os.mkdir("test/"+dir_name)
            shutil.copyfile(img_path, "test/"+dir_name+"/"+img_name)

def split_data(samples):
    split_rate = 0.6
    train_all = []
    test_all = []

    for class_id, img_in_class in enumerate(samples):
        rand_ind = [ i for i in xrange(len(img_in_class)) ]
        rand_train_ind = set(random.sample(rand_ind, int(len(img_in_class) * split_rate)))
        rand_test_ind = set(rand_ind) - rand_train_ind

        # train_in_class = []
        # test_in_class = []
        for ind in rand_train_ind:
            img_vec = img_in_class[ind]
            img_vec = img_vec.reshape(img_vec.shape[0] * img_vec.shape[1] * img_vec.shape[2],)
            train_all.append((class_id, img_vec))
        for ind in rand_test_ind:
            img_vec = img_in_class[ind]
            img_vec = img_vec.reshape(img_vec.shape[0] * img_vec.shape[1] * img_vec.shape[2],)
            test_all.append((class_id, img_vec))

        # train_all.append(train_in_class)
        # test_all.append(test_in_class)

    return train_all, test_all

def read_img(input_path):
    img_size = (200, 200)

    sample_all = []
    for dir_name in xrange(10):
        dir_name += 1
        dir_name = str(dir_name)

        dir_path = os.path.join(input_path, dir_name)
        img_in_class = []
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            img_vec = cv2.imread(img_path, flags=1)
            # print img_vec.shape
            # res = cv2.resize(img_vec, (int(img_vec.shape[0]*0.5), int(img_vec.shape[1]*0.5)), interpolation=cv2.INTER_CUBIC)
            res = cv2.resize(img_vec, img_size, interpolation=cv2.INTER_CUBIC)
            nor_res = np.zeros_like(res)
            nor_res = cv2.normalize(src=res, dst=nor_res, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            img_in_class.append(nor_res)

        sample_all.append(img_in_class)

    train_all, test_all = split_data(sample_all)
    return train_all, test_all

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

        img_in_class = []
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            img_vec = cv2.imread(img_path, flags=1)
            # print img_vec.shape
            # res = cv2.resize(img_vec, (int(img_vec.shape[0]*0.5), int(img_vec.shape[1]*0.5)), interpolation=cv2.INTER_CUBIC)
            res = cv2.resize(img_vec, img_size, interpolation=cv2.INTER_CUBIC)
            nor_res = np.zeros_like(res)
            nor_res = cv2.normalize(src=res, dst=nor_res, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            img_in_class.append(nor_res)

        train_all.append(img_in_class)

    # read test
    for dir_name in os.listdir(test_path):
        dir_path = os.path.join(test_path, dir_name)

        img_in_class = []
        for img_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, img_name)
            img_vec = cv2.imread(img_path, flags=1)
            # print img_vec.shape
            # res = cv2.resize(img_vec, (int(img_vec.shape[0]*0.5), int(img_vec.shape[1]*0.5)), interpolation=cv2.INTER_CUBIC)
            res = cv2.resize(img_vec, img_size, interpolation=cv2.INTER_CUBIC)
            nor_res = np.zeros_like(res)
            nor_res = cv2.normalize(src=res, dst=nor_res, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            img_in_class.append(nor_res)

        test_all.append(img_in_class)

    return train_all, test_all

if __name__ == "__main__":
    current_base = os.path.abspath('.')
    input_base = os.path.join(current_base, 'data')
    split_img(input_base)
    # train_all, test_all = read_data()
    # print train_all