import cv2
import numpy as np
from sklearn.utils import shuffle
import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tqdm import tqdm
class Prepare:
    def __init__(self):
        self.datagen = keras.preprocessing.image.ImageDataGenerator(
                                                                        rotation_range=8,
                                                                        width_shift_range=0.1,
                                                                        height_shift_range=0.1,
                                                                        shear_range=0.1,
                                                                        zoom_range=0.1,
                                                                        fill_mode='nearest',
                                                                        horizontal_flip=False,
                                                                        vertical_flip=False
                                                                    )

    def data(self,config):
        classes = config.data.classes
        training_images_dict = config.data.face_dict

        count = -1
        label_image = {}

        images = np.empty((0, 180, 180, 1))
        labels = np.empty(0, dtype='uint8')
        labels_number = np.empty(0, dtype='uint8')
        generate_total_images = 200

        for label in tqdm(classes):
            count = count + 1
            label_image[label] = cv2.imread("/labels/"+label+".jpg")
            images_c = []
            labels_c = []
            labels_number_c = []
            for image_per_label in training_images_dict[label]:
                # Smoothing and Variance Removal
                gray_scale_image = cv2.cvtColor(image_per_label, cv2.COLOR_BGR2GRAY)
                image_between_0_and_1 = gray_scale_image / 255.0
                image_between_0_and_1 = image_between_0_and_1 - 0.5
                normalized_image_between_ng_1_and_po_1 = image_between_0_and_1 * 2.0

                images_c.append(normalized_image_between_ng_1_and_po_1.reshape((config.image_resize_h, config.image_resize_w, 1)))
                labels_c.append(label)
                labels_number_c.append(count)

            images_c = np.asarray(images_c)
            labels_number_c = np.asarray(labels_number_c)

            #Images per class
            image_class_cpy = np.copy(images_c)
            label_number_class_cpy = np.copy(labels_number_c)

            for img, cls in self.datagen.flow(images_c, labels_number_c, batch_size=len(labels_number_c), seed=2 + count * 37):
                image_class_cpy = np.append(image_class_cpy, img, axis=0)
                label_number_class_cpy = np.append(label_number_class_cpy, cls, axis=0)

                if len(image_class_cpy) >= generate_total_images:
                    break

            images = np.append(images, image_class_cpy, axis=0)
            labels_number = np.append(labels_number, label_number_class_cpy, axis=0)

        shuffle_images, shuffle_labels_number = shuffle(np.array(images), np.array(labels_number))

        total_images = len(shuffle_images)
        training_sample_size = int(total_images * (int(config.training_size_percentage) / 100))

        training = {}
        training_sample_images = shuffle_images[0:training_sample_size]
        # training_sample_labels = shuffle_labels[0:training_sample_size]
        training_sample_labels_number = shuffle_labels_number[0:training_sample_size]
        training['images'] = training_sample_images
        training['labels'] = []
        training['labels_n'] = training_sample_labels_number

        testing = {}
        testing_sample_images = shuffle_images[training_sample_size:total_images]
        # testing_sample_labels = shuffle_labels[training_sample_size:total_images]
        testing_sample_labels_number = shuffle_labels_number[training_sample_size:total_images]
        testing['images'] = testing_sample_images
        testing['labels'] = []
        testing['labels_n'] = testing_sample_labels_number



        config.data.train_data = training
        config.data.test_data = testing
        config.data.label_image = label_image

        print("Training Image Size : ", str(len(training_sample_images)),str(len(training_sample_labels_number)))
        print("Testing Image Size : ", str(len(testing_sample_images)), str(len(testing_sample_labels_number)))
