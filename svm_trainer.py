import cv2
from skimage.feature import hog

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import glob
import numpy as np
import matplotlib.pyplot as plt


def color_hist(img, nbins=64, bins_range=(0, 256)):
    # Compute the histogram of the image
    hist_features = np.histogram(img, bins=nbins, range=bins_range)
    # Return feature vector
    return hist_features[0]

def extract_hog(img, pix_per_cell=6, cell_per_block=2, orient=12, vis=False):
    # Container to store features and visualized image
    hog_features = []
    hog_feature = hog(img, orientations=orient,
                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                      cells_per_block=(cell_per_block, cell_per_block), 
                      visualize=vis)
    # Append the new feature vector to the features list
    hog_features.append(hog_feature)
    
    # Return hog feature vector
    return np.concatenate(hog_features)

def extract_features(image):

    # Read in image
    if type(image) == str:
        image = cv2.imread(image)
    # Resize
    if image[:, :, 0].shape != (64, 64):
        image = cv2.resize(image, (64, 64), cv2.INTER_AREA)

    # Convert image to HSV space
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Exact features
    #color_features = hsv[:,:,0].flatten()
    hist_features = color_hist(hsv[:, :, 0])
    hog_features = extract_hog(gray)

    features = np.concatenate((hist_features, hog_features))
    return features


def visualize_features():
    # Load two samples
    stop_list = glob.glob("svm_training_dataset/true_set/*.png")
    non_stop_list = glob.glob("svm_training_dataset/false_set/*.png")
    stop_test = cv2.imread(stop_list[100])
    non_stop_test = cv2.imread(non_stop_list[50])
    images = [stop_test, non_stop_test]

    fig, axes = plt.subplots(len(images), 3, figsize=(18, 3*len(images)))
    for idx, image in enumerate(images):
        images = cv2.resize(image, (64, 64), cv2.INTER_AREA)
        # Convert to HSV and YUV space
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # Extract HSV features
        h_features = color_hist(hsv[:, :, 0])
        # Extract YUV HOG features
        yuv_hog_features, yuv_hog_images = extract_hog(gray, vis=True)
        # Display
        axes[idx, 0].set_title("Origin")
        axes[idx, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[idx, 1].set_title("HSV: H histogram")
        axes[idx, 1].plot(h_features)
        axes[idx, 2].set_title("Gray: HOG")
        axes[idx, 2].imshow(yuv_hog_images, cmap='gray')

    plt.show()


def train_svm():
    # Images data path
    stop_list = glob.glob("svm_training_dataset/true_set/*.png")
    non_stop_list = glob.glob("svm_training_dataset/false_set/*.png")
    
    # Exact images' features
    stop_features = [extract_features(image) for image in stop_list]
    non_stop_features = [extract_features(image) for image in non_stop_list]

    # Create an array stack of feature vectors
    X = np.vstack((stop_features, non_stop_features)).astype(np.float64)
    # Define the labels vector
    y = np.hstack((np.ones(len(stop_features)), np.zeros(len(non_stop_features))))

    # Split up data into randomized training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
                                            X, y, test_size=0.2, random_state=10)

    # Normalization
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)


    # Use a linear SVC
    svc = LinearSVC()
    # Train the SCV
    svc.fit(X_train, y_train)
    
    # Save trained SVM
    joblib.dump((svc, X_scaler), "svm_model.m")

    # Test
    accuracy = svc.score(X_test, y_test)
    print('Test accuracy of SVM = ', accuracy)


if __name__ == "__main__":
    train_svm()
    #visualize_features()
    