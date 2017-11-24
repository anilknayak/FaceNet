# Load Data

import pickle
import os


class LoadData:
    def __init__(self):
        ''

    def data(self):
        training_images_pickle = open("training_images.pickle", "rb")
        training_images_pickle_file_dict = pickle.load(training_images_pickle)
        classes = training_images_pickle_file_dict['classes']
        classes_number = len(training_images_pickle_file_dict['classes'])
        return training_images_pickle_file_dict, classes, classes_number

    def find_classes(self,training_folder):
        classes = []
        classes_count = []
        files = os.listdir(training_folder)
        count = 1
        for dirs in files:
            if os.path.isdir(os.path.join(training_folder, dirs)):
                classes.append(dirs)
                classes_count.append(count)
                count = count + 1

        return classes, classes_count

