import cv2 
import matplotlib.pyplot as plt 
import time

from data_reader import DataReader


def pre_processing(img):
    img = cv2.medianBlur(img.copy())
    
    
    gray_img = None

    return gray_img

def extra_area(gray_img):
    images = []
    return images


def svm_classify(img):
    result = []
    return result


def hsv_svm(img):
    '''
    Parameters:
    gray_img (2d numpy array): Input gray scale image

    Returns:
    a list of found stop signs areas [(x, y, height, weight), ...]
    if nothing is found, return an empty list []
    '''
    # Extract ROI
    gray_img = pre_processing(img)
    images = extra_area(gray_img)
    # Check each possible region
    found = []
    for image in images:
        result = svm_classify(image)
        if result is not None:
            found.append(found)

    return found


def main():
    data = DataReader()

    # Test with LISA dataset
    data.load_dataset("lisa")
    tick = time.time()
    accuracy = data.run_test(hsv_svm, color=True)
    print("The running time is ", time.time()-tick)
    print("The accuracy of running LISA dataset is: ", accuracy)
    print("Accurate Indices: ", data.accurate_indices)
    data.visualize_result()

    # Test with online dataset
    data.load_dataset("online")
    accuracy = data.run_test(hsv_svm, color=True)
    print("The running time is ", time.time()-tick)
    print("The accuracy of running online dataset is: ", accuracy)
    print("Accurate Indices: ", data.accurate_indices)
    data.visualize_result()


if __name__ == "__main__":
    #main()
    cv2.imread("lisa_dataset/images/stop_1323804701.avi_image3.png")
