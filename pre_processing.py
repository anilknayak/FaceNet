import imutils
import dlib
import os
import load_data as ld
import data as data
import cv2
from imutils import face_utils
import pickle
import prepare_data_training as pdt

class PreProcessing:
    def __init__(self):
        self.dlib_model = "dlib_pretrained_model.dat"
        self.load = ld.LoadData()
        self.prepare = pdt.Prepare()
        self.prepare = pdt.Prepare()

    def prepare_pre_procesing_folder_structure(self,config):
        files = os.listdir(config.image_directory)
        load = ld.LoadData()
        pre_image_processing_path = os.path.join(config.image_directory, "data")
        config.pre_processing_folder = pre_image_processing_path
        classes, classes_count = self.load.find_classes(config.pre_processing_folder)

        post_image_processing_path = os.path.join(config.image_directory, "data_post_processing")

        if not os.path.exists(post_image_processing_path):
            os.mkdir(post_image_processing_path)
            config.post_processing_folder = post_image_processing_path
        else:
            config.post_processing_folder = post_image_processing_path

        dt = data.Data()
        dt.classes = classes
        dt.classes_count = classes_count

        config.data = dt


    def prepare_faces_from_training_images_for_training(self,config):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(self.dlib_model)
        count = -1
        for label in config.data.classes:
            count = count + 1
            label_path = os.path.join(config.pre_processing_folder, label)
            face_label_path = os.path.join(config.post_processing_folder, label)

            if not os.path.exists(face_label_path):
                os.mkdir(face_label_path)

            images_per_labels = os.listdir(label_path)

            number_of_faces = 0
            for image in images_per_labels:
                image_path = os.path.join(label_path, image)
                image = cv2.imread(image_path)
                rects = detector(image, 1)
                for rect in rects:
                    number_of_faces = number_of_faces + 1
                    (x, y, w, h) = face_utils.rect_to_bb(rect)
                    face = image[y - 50: y + h + 10, x - 10: x + w + 20]
                    face = cv2.resize(face, (config.image_resize_w, config.image_resize_h), interpolation=cv2.INTER_CUBIC)
                    face_image_file_name = label + "_" + str(config.data.classes_count[count]) + "_" + str(number_of_faces) + ".jpg"
                    file_name = os.path.join(face_label_path, face_image_file_name)
                    cv2.imwrite(file_name, face)

        classes, classes_count = self.load.find_classes(config.post_processing_folder)
        config.data.classes = classes
        config.data.classes_count = classes_count

    def prepare_training_dictionary(self,folder,config):
        face_dictionary = {}

        for label in config.data.classes:
            label_path = os.path.join(folder, label)
            images_per_labels = os.listdir(label_path)
            images = []
            for image in images_per_labels:
                image_path = os.path.join(label_path, image)
                image = cv2.imread(image_path)
                images.append(image)

            face_dictionary[label] = images

        config.data.face_dict = face_dictionary

    def prepare_training_testing_data(self,config):
        self.prepare.data(config)

    def prepare_pickle_file(self,config):
        config.data.pickle_file = config.pickle_data_file
        face_data = {}
        face_data['label_image'] = config.data.label_image
        face_data['train'] = config.data.train_data
        face_data['test'] = config.data.test_data
        face_data['classes'] = config.data.classes
        face_data['classes_n'] = config.data.classes_count
        pickle_in = open(config.data.pickle_file, "wb")
        pickle.dump(face_data, pickle_in)
        pickle_in.close()






