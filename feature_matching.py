import cv2
import numpy as np
import math
import time

from data_reader import DataReader


protoimg = cv2.imread('demo/stop_sign_model6.jpg')#, cv2.IMREAD_GRAYSCALE
protoimg = cv2.resize(protoimg, None, protoimg, 0.7, 0.7)
# protoimg = cv2.medianBlur(protoimg, 5)

def getAngleABC(pta, ptb, ptc):
    ab = (ptb[0] - pta[0], ptb[1] - pta[1])
    cb = (ptb[0] - ptc[0], ptb[1] - ptc[1])

    dot = (ab[0] * cb[0] + ab[1] * cb[1])
    cross = (ab[0] * cb[1] - ab[1] * cb[0])
    alpha = math.atan2(cross, dot)

    return alpha * 180 / np.pi

def pre_processing(img):
    # img = cv2.medianBlur(img, 5)
    
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
    r_mask = cv2.morphologyEx(r_in_range, cv2.MORPH_DILATE, np.ones((5,5), np.uint8))
    
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
        if area < 150:
            continue
        # ratio verification
        ratio = rect[2] / rect[3]
        if ratio > 10 or ratio < 0.1:
            continue

        # store image and coordinates (x, y, height, width)
        sliced_img = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

        # cv2.imshow(" ", sliced_img)
        # cv2.waitKey(0)

        images.append(sliced_img)
        coords.append(rect)

    return images, coords

def feature_matching(gray_img):
    '''
    Parameters:
    gray_img (2d numpy array): Input gray scale image

    Returns:
    a list of found stop signs areas [(x, y, height, weight) ...]
    if nothing is found, return an empty list []
    '''
    found = []

    fx = 900 / gray_img.shape[1]
    gray_img = cv2.resize(gray_img, None, gray_img, fx, fx)
    # gray_img = cv2.medianBlur(gray_img, 5)

    sift = cv2.SIFT_create()
    kp2, des2 = sift.detectAndCompute(protoimg, None)

    b_img = pre_processing(gray_img)
    # cv2.imshow('11', b_img)
    # cv2.waitKey()
    images, coords = extract_area(gray_img, b_img)
    for image, coord in zip(images, coords):
        kp1, des1 = sift.detectAndCompute(image, None)
        # cv2.imshow('11', image)
        # cv2.waitKey()

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        # matcher = cv2.BFMatcher()
        # raw_matches = matcher.knnMatch(des1, des2, k = 2)
        # print(len(raw_matches))

        good_matches = []
        for m1, m2 in raw_matches:
            if m1.distance < 0.9 * m2.distance:
                good_matches.append([m1])

        # img_out = cv2.drawMatchesKnn(gray_img, kp1, protoimg, kp2, good_matches, None, flags=2)
        # img_out = cv2.drawMatchesKnn(image, kp1, protoimg, kp2, good_matches, None, flags=2)
        # cv2.imshow('feature matching', img_out)
        # cv2.waitKey()

        if len(good_matches) > 5:
            # src_pts = np.asarray([kp1[m[0].queryIdx].pt for m in good_matches])
            # dst_pts = np.asarray([kp2[m[0].trainIdx].pt for m in good_matches])

            # Constrain matches to fit homography
            # retval, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            # if retval is None:
            #     # return found
            #     continue
            # mask = mask.ravel()

            # h,w,d = protoimg.shape
            # box = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            # box_det = cv2.perspectiveTransform(box, np.linalg.inv(retval))
            
            # img_det = cv2.polylines(gray_img,[np.int32(box_det)], True, 255, 2, cv2.LINE_AA)
            # img_det = cv2.polylines(image,[np.int32(box_det)], True, 255, 3, cv2.LINE_AA)
            # cv2.imshow('result', img_det)
            # cv2.waitKey()

            # out_box_tmp = np.squeeze(box_det)
            # for e in out_box_tmp:
            #     if e[0] < 0 or e[1] < 0 or e[0] > gray_img.shape[1] or e[1] > gray_img.shape[0]:
            #         # return found 
            #         continue
            # if out_box_tmp[2][0] < out_box_tmp[0][0] or out_box_tmp[2][1] < out_box_tmp[0][1]:
            #     # return found
            #     continue
            
            # angle1 = getAngleABC(out_box_tmp[0], out_box_tmp[1], out_box_tmp[2])
            # angle2 = getAngleABC(out_box_tmp[1], out_box_tmp[2], out_box_tmp[3])
            # angle3 = getAngleABC(out_box_tmp[2], out_box_tmp[3], out_box_tmp[1])
            # if angle1 < 20 or angle2 < 20 or angle3 < 20:
            #     # return found
            #     continue

            # print(1)
            
            # Display
            # matchesMask = [[i, i] for i in mask.tolist()]
            # draw_params = dict(matchColor = (0,0,255),
            #                     singlePointColor = (255,0,0),
            #                     matchesMask = matchesMask,
            #                     flags = 0)
            
            # matches = cv2.drawMatchesKnn(cv2.cvtColor(gray_img, cv2.COLOR_BGR2RGB), kp1, 
            #                             cv2.cvtColor(protoimg, cv2.COLOR_BGR2RGB), kp2, 
            #                             good_matches, None, **draw_params)
            
            # cv2.imshow('1', matches)
            
            # xmin = np.min(out_box_tmp[:, 0])
            # xmax = np.max(out_box_tmp[:, 0])
            # ymin = np.min(out_box_tmp[:, 1])
            # ymax = np.max(out_box_tmp[:, 1])
            # # out = (np.int32(xmin), np.int32(ymin), np.int32((ymax - ymin)), np.int32((xmax - xmin)))
            # out = (np.int32(xmin / fx), np.int32(ymin / fx), np.int32((ymax - ymin) / fx), np.int32((xmax - xmin) / fx))
            # found.append(out)
            out = (np.int32(coord[0] / fx), np.int32(coord[1] / fx), np.int32(coord[2] / fx), np.int32(coord[3] / fx))
            found.append(out)
            # print(out)
            # print(out_box_tmp)
        else:
            # img_det = cv2.polylines(gray_img,[np.int32(box_det)], True, 255, 3, cv2.LINE_AA)
            # cv2.imshow('result', gray_img)
            # cv2.waitKey()
            pass
    
    # for (x, y, height, width) in found: 
    #     cv2.rectangle(gray_img, (int(x*fx), int(y*fx)), 
    #                                     (int(x*fx + height*fx), int(y*fx + width*fx)),  
    #                     (0, 255, 0), 5)
    # cv2.imshow('2', gray_img)
    # cv2.waitKey()

    return found


def main():
    data = DataReader()

    # Test with LISA dataset
    data.load_dataset("lisa")
    tick = time.time()
    accuracy = data.run_test(feature_matching, True)
    print("The running time is ", time.time()-tick)
    print("The accuracy of running LISA dataset is: ", accuracy)
    print("Accurate Indices: ", data.accurate_indices)
    data.visualize_result()

    # Test with online dataset
    data.load_dataset("online")
    tick = time.time()
    accuracy = data.run_test(feature_matching)
    print("The running time is ", time.time()-tick)
    print("The accuracy of running online dataset is: ", accuracy)
    print("Accurate Indices: ", data.accurate_indices)
    data.visualize_result()


if __name__ == "__main__":
    main()
    