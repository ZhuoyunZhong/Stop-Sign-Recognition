import cv2 
import numpy as np
import matplotlib.pyplot as plt 
import time

import joblib
from data_reader import DataReader
from svm_trainer import extract_features


def pre_processing(img):
    img = cv2.medianBlur(img, 5)
    # HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red color range to find the traffic sign
    low_r_range1 = np.array([0, 70, 30])
    up_r_range1 = np.array([10, 255, 255])
    low_r_range2 = np.array([160, 70, 30])
    up_r_range2 = np.array([180, 255, 255])
    # Find the area in range
    r_in_range1 = cv2.inRange(hsv_img, low_r_range1, up_r_range1)
    r_in_range2 = cv2.inRange(hsv_img, low_r_range2, up_r_range2)
    r_in_range = cv2.bitwise_or(r_in_range1, r_in_range2)

    # Binary morphological operation
    r_mask = cv2.morphologyEx(r_in_range, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))

    return r_mask
    

def extract_area(img, binary_img):
    images = []
    coords = []

    # Find contours in the mask
    conts = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    # For each contour
    c = 0
    for cont in conts:
        # bounding box
        rect = cv2.boundingRect(cont)

        # area verification
        area = rect[2] * rect[3] 
        if area < 200:
            continue
        # ratio verification
        ratio = rect[2] / rect[3]
        if ratio > 3 or ratio < 0.333:
            continue

        # store image and coordinates (x, y, height, width)
        coord = list(rect)
        coord[0] = coord[0]-10 if coord[0]-10>0 else 0
        coord[1] = coord[1]-10 if coord[1]-10>0 else 0
        coord[2] += 20
        coord[3] += 20 
        sliced_img = img[coord[1]:coord[1]+coord[3], coord[0]:coord[0]+coord[2]]
        sliced_img = cv2.resize(sliced_img, (64, 64), cv2.INTER_AREA)
        images.append(sliced_img)
        coords.append(coord)

    return images, coords


def svm_classify(svm, scaler, img):
    # Feature extraction
    features = extract_features(img)
    test_features = scaler.transform(np.array(features).reshape(1, -1))

    # Predict using your SVM
    prediction = svm.predict(test_features)
    if prediction == 1:
        return True
    else:
        return False


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
    svm, scaler = joblib.load("svm_model.m")
    for i, image in enumerate(images):
        result = svm_classify(svm, scaler, image)
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
    #for i in range(100):
    #    data.visualize_result(i, write=True)
    
    # Test with online dataset
    data.load_dataset("online")
    tick = time.time()
    accuracy = data.run_test(hsv_svm, color=True)
    print("The running time is ", time.time()-tick)
    print("The accuracy of running online dataset is: ", accuracy)
    print("Accurate Indices: ", data.accurate_indices)
    #for i in range(200):
    #    data.visualize_result(i)

if __name__ == "__main__":
    main()
