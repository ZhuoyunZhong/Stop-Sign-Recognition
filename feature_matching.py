import cv2
import numpy as np
import math
import time

from data_reader import DataReader


protoimg = cv2.imread('demo/stop_sign_model1.jpg', cv2.IMREAD_GRAYSCALE)
protoimg = cv2.resize(protoimg, None, protoimg, 0.7, 0.7)


'''
int getAngleABC( cv::Point2d a, cv::Point2d b, cv::Point2d c )
{
    cv::Point2d ab = { b.x - a.x, b.y - a.y };
    cv::Point2d cb = { b.x - c.x, b.y - c.y };

    float dot = (ab.x * cb.x + ab.y * cb.y); // dot product
    float cross = (ab.x * cb.y - ab.y * cb.x); // cross product

    float alpha = atan2(cross, dot);

    return (int) floor(alpha * 180. / M_PI + 0.5);
}
'''

def getAngleABC(pta, ptb, ptc):
    ab = (ptb[0] - pta[0], ptb[1] - pta[1])
    cb = (ptb[0] - ptc[0], ptb[1] - ptc[1])

    dot = (ab[0] * cb[0] + ab[1] * cb[1])
    cross = (ab[0] * cb[1] - ab[1] * cb[0])
    alpha = math.atan2(cross, dot)

    return alpha * 180 / np.pi


def feature_matching(gray_img):
    '''
    Parameters:
    gray_img (2d numpy array): Input gray scale image

    Returns:
    a list of found stop signs areas [(x, y, width, height) ...]
    if nothing is found, return an empty list []
    '''
    found = []

    # cv2.imshow('1', protoimg)
    # cv2.waitKey()
    
    fx = 900 / gray_img.shape[1]
    # print(fx)
    gray_img = cv2.resize(gray_img, None, gray_img, fx, fx)
    
    sift = cv2.SIFT_create()
    kp2, des2 = sift.detectAndCompute(protoimg, None)
    kp1, des1 = sift.detectAndCompute(gray_img, None)
    

    # kp_image1 = cv2.drawKeypoints(gray_img, kp1, None)
    # kp_image2 = cv2.drawKeypoints(protoimg, kp2, None)
    
    # cv2.imshow('1', kp_image1)
    # cv2.imshow('2', kp_image2)
    # cv2.waitKey()

    # matcher = cv2.FlannBasedMatcher_create()
    # raw_matches = matcher.knnMatch(des1, des2, k=2)
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
        if m1.distance < 0.7 * m2.distance:
            good_matches.append([m1])

    if len(good_matches) > 8:
        src_pts = np.asarray([kp1[m[0].queryIdx].pt for m in good_matches])
        dst_pts = np.asarray([kp2[m[0].trainIdx].pt for m in good_matches])

        # Constrain matches to fit homography
        retval, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if retval is None:
            return found
        mask = mask.ravel()

        h,w = protoimg.shape
        box = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        box_det = cv2.perspectiveTransform(box, np.linalg.inv(retval))
        
        out_box_tmp = np.squeeze(box_det)
        for e in out_box_tmp:
            if e[0] < 0 or e[1] < 0 or e[0] > gray_img.shape[1] or e[1] > gray_img.shape[0]:
                return found 
        if out_box_tmp[2][0] < out_box_tmp[0][0] or out_box_tmp[2][1] < out_box_tmp[0][1]:
            return found
        
        angle1 = getAngleABC(out_box_tmp[0], out_box_tmp[1], out_box_tmp[2])
        angle2 = getAngleABC(out_box_tmp[1], out_box_tmp[2], out_box_tmp[3])
        angle3 = getAngleABC(out_box_tmp[2], out_box_tmp[3], out_box_tmp[1])
        # print(angle1, angle2, angle3)
        if angle1 < 20 or angle2 < 20 or angle3 < 20:
            return found
        
        # img_det = cv2.polylines(gray_img,[np.int32(box_det)], True, 255, 3, cv2.LINE_AA)
        # cv2.imshow('result', img_det)


        # We select only inlier points
        # pts1 = src_pts[mask == 1]
        # pts2 = dst_pts[mask == 1]

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
        

        xmin = np.min(out_box_tmp[:, 0])
        xmax = np.max(out_box_tmp[:, 0])
        ymin = np.min(out_box_tmp[:, 1])
        ymax = np.max(out_box_tmp[:, 1])
        out = (np.int32(xmin / fx), np.int32(ymin / fx), np.int32((ymax - ymin) / fx), np.int32((xmax - xmin) / fx))
        found.append(out)
        # print(out)
        # print(out_box_tmp)
        # cv2.waitKey()



    # matches = cv2.drawMatchesKnn(gray_img, kp1, protoimg, kp2, good_matches, None, flags = 2)
    
    # if matches.shape[1] > 1800:
    # matches = cv2.resize(matches, None, matches, 0.5, 0.5)

    # cv2.imshow('1', matches)
    # cv2.imshow('2', kp_image2)
    # cv2.waitKey()

    return found


def main():
    data = DataReader()

    # Test with LISA dataset
    data.load_dataset("lisa")
    tick = time.time()
    accuracy = data.run_test(feature_matching)
    print("The running time is ", time.time()-tick)
    print("The accuracy of running LISA dataset is: ", accuracy)
    print("Accurate Indices: ", data.accurate_indices)
    data.visualize_result()

    # Test with online dataset
    data.load_dataset("online")
    accuracy = data.run_test(feature_matching)
    print("The running time is ", time.time()-tick)
    print("The accuracy of running online dataset is: ", accuracy)
    print("Accurate Indices: ", data.accurate_indices)
    data.visualize_result()


if __name__ == "__main__":
    main()
    