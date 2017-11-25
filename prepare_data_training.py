import cv2
import numpy as np
from sklearn.utils import shuffle


class Prepare:
    def __init__(self):
        ''

    def data(self,config):
        classes = config.data.classes
        training_images_dict = config.data.face_dict
        images = []
        labels = []
        labels_number = []
        count = -1
        label_image = {}
        for label in classes:
            count = count + 1
            label_image[label] = cv2.imread("/labels/"+label+".jpg")
            for image_per_label in training_images_dict[label]:
                # Smoothing and Variance Removal
                gray_scale_image = cv2.cvtColor(image_per_label, cv2.COLOR_BGR2GRAY)
                image_between_0_and_1 = gray_scale_image / 255.0
                image_between_0_and_1 = image_between_0_and_1 - 0.5
                normalized_image_between_ng_1_and_po_1 = image_between_0_and_1 * 2.0

                images.append(normalized_image_between_ng_1_and_po_1.reshape((config.image_resize_h, config.image_resize_w, 1)))
                labels.append(label)
                labels_number.append(count)

        shuffle_images, shuffle_labels, shuffle_labels_number = shuffle(np.array(images), np.array(labels),
                                                                        np.array(labels_number))

        total_images = len(shuffle_images)
        training_sample_size = int(total_images * (int(config.training_size_percentage) / 100))

        training = {}
        training_sample_images = shuffle_images[0:training_sample_size]
        training_sample_labels = shuffle_labels[0:training_sample_size]
        training_sample_labels_number = shuffle_labels_number[0:training_sample_size]
        training['images'] = training_sample_images
        training['labels'] = training_sample_labels
        training['labels_n'] = training_sample_labels_number

        testing = {}
        testing_sample_images = shuffle_images[training_sample_size:total_images]
        testing_sample_labels = shuffle_labels[training_sample_size:total_images]
        testing_sample_labels_number = shuffle_labels_number[training_sample_size:total_images]
        testing['images'] = testing_sample_images
        testing['labels'] = testing_sample_labels
        testing['labels_n'] = testing_sample_labels_number

        config.data.train_data = training
        config.data.test_data = testing
        config.data.label_image = label_image
