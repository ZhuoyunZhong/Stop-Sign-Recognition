import cv2 
import numpy as np
import matplotlib.pyplot as plt 
import time

from data_reader import DataReader


def pre_processing(img):
    img = cv2.medianBlur(img, 5)
    
    # HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red color range to find the traffic sign
    #low_b_range = np.array([90, 100, 70])
    #up_b_range = np.array([130, 255, 255])
    low_r_range1 = np.array([0, 100, 70])
    up_r_range1 = np.array([10, 255, 255])
    low_r_range2 = np.array([160, 100, 70])
    up_r_range2 = np.array([180, 255, 255])
    # Find the area in range
    #b_in_range = cv2.inRange(hsv_img, low_b_range, up_b_range)
    r_in_range1 = cv2.inRange(hsv_img, low_r_range1, up_r_range1)
    r_in_range2 = cv2.inRange(hsv_img, low_r_range2, up_r_range2)
    r_in_range = cv2.bitwise_or(r_in_range1, r_in_range2)

    # Binary morphological operation
    #b_mask = cv2.morphologyEx(b_in_range, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    r_mask = cv2.morphologyEx(r_in_range, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    
    return r_mask
    

def extract_area(img, binary_img):
    images = []
    coords = []

    # Find contours in the mask
    conts = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    # For each contour
    for cont in conts:
        # bounding box
        rect = cv2.boundingRect(cont)

        # area verification
        area = rect[2] * rect[3] 
        if area < 100:
            continue
        # ratio verification
        ratio = rect[2] / rect[3]
        if ratio > 10 or ratio < 0.1:
            continue

        # store image and coordinates (x, y, height, width)
        sliced_img = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

        cv2.imshow(" ", sliced_img)
        cv2.waitKey(0)

        images.append(sliced_img)
        coords.append(rect)

    return images, coords


def svm_classify(img):
    result = []
    return True


def hsv_svm(img):
    '''
    Parameters:
    gray_img (2d numpy array): Input gray scale image

    Returns:
    a list of found stop signs areas [(x, y, height, weight), ...]
    if nothing is found, return an empty list []
    '''
    # Extract ROI
    binary_img = pre_processing(img)
    images, coords = extract_area(img, binary_img)
    # Check each possible region
    found = []
    for i, image in enumerate(images):
        result = svm_classify(image)
        if result:
            found.append(coords[i])

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
    img = cv2.imread("demo/stop_sign.jpg")
    #cv2.imshow("img", img)
    #cv2.waitKey(0)
    print(hsv_svm(img))
