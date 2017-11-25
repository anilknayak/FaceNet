# Load Data

import pickle
import os
import data as data

class LoadData:
    def __init__(self):
        ''

    def data(self,config):
        training_images_pickle = open(config.pickle_data_file, "rb")
        training_images_pickle_file_dict = pickle.load(training_images_pickle)
        classes = training_images_pickle_file_dict['classes']
        classes_number = training_images_pickle_file_dict['classes_n']
        test_data = training_images_pickle_file_dict['test']
        train_data = training_images_pickle_file_dict['train']


        dt = data.Data()
        dt.classes = classes
        dt.classes_count = classes_number
        dt.train_data = train_data
        dt.test_data = test_data
        config.data = dt


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

