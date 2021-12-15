import cv2
import os
# Classes and functions to help reading the dataset


class LabeledImage():
    # Test image
    def __init__(self):
        self.image = None
        self.gray_image = None
        self.label = None

    def set_image(self, img):
        # save image
        if type(img) is str:
            self.image = cv2.imread(img)
        else:
            self.image = img
        # save gray scale image
        if len(self.image.shape) == 2:
            self.gray_image = self.image.copy()
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        else:
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def set_label(self, label):
        # set label
        self.label = label

    def is_stop_sign(self):
        # return if this is a stop sign image
        return self.label == "stop"


class DataReader():
    # Read test images and their labels
    def __init__(self):
        self.initialize_dataset()

    def initialize_dataset(self):
        # Storage images
        self.labeled_images = []
        self.labels = []
        self.are_stop_signs = []
        # Storage test results
        self.results = []
        self.accurate_indices = []
        self.accurate_stop_sign_indices = []
        self.found_areas = []


    def load_dataset(self, name="lisa"):
        self.initialize_dataset()
        if name == "lisa":
            self.load_lisa()
        else:
            self.load_online()

    def load_lisa(self):
        # Initialization
        # Read images from LISA dataset
        folder_name = "lisa_dataset/images"
        image_names = os.listdir(folder_name)
        for image_name in image_names:
            image = LabeledImage()
            image.set_image(folder_name+"/"+image_name)
            label = image_name.split("_")[0]
            image.set_label(label)
            # Save
            self.labels.append(label)
            self.labeled_images.append(image)
        # Generate a new list for checking accuracy
        self.generate_stop_sign_flags()

    def load_online(self):
        # Initialization
        
        # Read images from LISA dataset
        folder_name = "online_dataset/images"
        image_names = os.listdir(folder_name)
        for image_name in image_names:
            image = LabeledImage()
            image.set_image(folder_name+"/"+image_name)
            file_name = image_name.split(".")[0]
            label = "stop" if int(file_name) < 100 else "others"
            image.set_label(label)
            # Save
            self.labels.append(label)
            self.labeled_images.append(image)
        # Generate a new list for checking accuracy
        self.generate_stop_sign_flags()

    def generate_stop_sign_flags(self):
        self.are_stop_signs = [image.is_stop_sign() 
                               for image in self.labeled_images]        


    def compute_accuracy(self, labels):
        # True false comparison
        result = [i==j for i, j in zip(self.are_stop_signs, labels)]
        
        # Get the indices that are true and are "stop" labels
        self.accurate_indices = [i for i, x in enumerate(result) if x]
        self.accurate_stop_sign_indices = [i for i in self.accurate_indices 
                                           if self.labeled_images[i].is_stop_sign()]

        # Compute Accuracy
        count = result.count(True)
        return float(count) / len(result)


    def run_test(self, classifier_function, color=False):
        # Run to test the classifier_function
        for image in self.labeled_images:
            if not color:
                found_areas = classifier_function(image.gray_image)
            else:
                found_areas = classifier_function(image.image)
            self.results.append(len(found_areas) != 0)
            self.found_areas.append(found_areas)

        # Get accuracy
        accuracy = self.compute_accuracy(self.results)
        return accuracy

    
    def visualize_result(self, index=None, write=False):
        # Visualize the first accurate stop sign if not specified
        if index == None:
            if len(self.accurate_stop_sign_indices) > 0:
                index = self.accurate_stop_sign_indices[0]
            else:
                index = 0
        # Load image and areas
        display_image = self.labeled_images[index].image.copy()
        found = self.found_areas[index]
        for (x, y, height, width) in found: 
            # Draw a green rectangle around every recognized sign 
            cv2.rectangle(display_image, (x, y), 
                                         (x + height, y + width),  
                          (0, 255, 0), 5)
        # Display
        if not write:
            cv2.imshow("Detection Result", display_image) 
            cv2.waitKey(0)
        else:
            cv2.imwrite("result/"+str(index)+".jpg", display_image)
