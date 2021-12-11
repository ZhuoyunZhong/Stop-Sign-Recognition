import cv2
import numpy as np

from data_reader import DataReader


def feature_matching(gray_img):
    '''
    Parameters:
    gray_img (2d numpy array): Input gray scale image

    Returns:
    a list of found stop signs areas [(x, y, width, height) ...]
    if nothing is found, return an empty list []
    '''
    found = []
    return found


def main():
    data = DataReader()

    # Test with LISA dataset
    data.load_dataset("lisa")
    accuracy = data.run_test(feature_matching)
    print("The accuracy of running LISA dataset is: ", accuracy)
    print("Accurate Indices: ", data.accurate_indices)
    data.visualize_result()

    # Test with online dataset
    data.load_dataset("online")
    accuracy = data.run_test(feature_matching)
    print("The accuracy of running online dataset is: ", accuracy)
    print("Accurate Indices: ", data.accurate_indices)
    data.visualize_result()


if __name__ == "__main__":
    main()
    