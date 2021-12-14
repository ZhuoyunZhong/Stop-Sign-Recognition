import cv2 
import matplotlib.pyplot as plt 
import time

from data_reader import DataReader

# https://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset


def haar_cascade(img_gray):
    '''
    Parameters:
    gray_img (2d numpy array): Input gray scale image

    Returns:
    a list of found stop signs areas [(x, y, height, weight), ...]
    if nothing is found, return an empty list []
    '''
    found = []
    x, y, height, weight = [], [], [], []
    stop_data = cv2.CascadeClassifier('cascade.xml')
    results = stop_data.detectMultiScale(img_gray,minNeighbors = 12) 
    if(len(results) != 0): 
        for i in results:
            found.append(i)
    return found

    

def main():
    data = DataReader()

    # Test with LISA dataset
    data.load_dataset("lisa")
    tick = time.time()
    accuracy = data.run_test(haar_cascade)
    print("The running time is ", time.time()-tick)
    print("The accuracy of running LISA dataset is: ", accuracy)
    print("Accurate Indices: ", data.accurate_indices)
    data.visualize_result()

    # Test with online dataset
    data.load_dataset("online")
    accuracy = data.run_test(haar_cascade)
    print("The running time is ", time.time()-tick)
    print("The accuracy of running online dataset is: ", accuracy)
    print("Accurate Indices: ", data.accurate_indices)
    data.visualize_result()


if __name__ == "__main__":
    main()
