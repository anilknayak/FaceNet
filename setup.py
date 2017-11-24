import json
import configuration as cfg
import pre_processing as pp
import prepare_network as pn

class SetUp:
    def __init__(self):
        self.configuration = cfg.Configuration()
        self.network = pn.Network()

        with open("face_recognition.config") as configuration:
            configuration_data = configuration.readlines()
            configuration_details = ""
            for line in configuration_data:
                configuration_details += line
            configuration_details_json = json.loads(str(configuration_details))

            self.configuration.network = configuration_details_json['network_config']['network']
            self.configuration.image_w = configuration_details_json['image']['width']
            self.configuration.image_h = configuration_details_json['image']['height']
            self.configuration.training_size_percentage = configuration_details_json['training']['training_size_percentage']
            self.configuration.image_resize_w = configuration_details_json['image']['width']
            self.configuration.image_resize_h = configuration_details_json['image']['height']
            self.configuration.learning_rate = configuration_details_json['training']['learning_rate']
            self.configuration.batch_size = configuration_details_json['training']['batch_size']
            self.configuration.training_steps = configuration_details_json['training']['training_steps']
            self.configuration.random_shuffle = configuration_details_json['training']['random_shuffle']
            self.configuration.training_folder = configuration_details_json['training']['training_data_folder']
            self.configuration.testing_folder = configuration_details_json['training']['training_data_folder']
            self.configuration.pre_processing_required = configuration_details_json['pre_processing_required']
            self.configuration.base_directory = configuration_details_json['training']['base_directory']
            self.configuration.image_directory = configuration_details_json['training']['image_directory']

            if configuration_details_json['network_config']['input_size'] == 'auto':
                self.configuration.network[0]['width'] = self.configuration.image_w
                self.configuration.network[0]['height'] = self.configuration.image_h


            number_of_classes = 0

            preproc = pp.PreProcessing()
            if self.configuration.pre_processing_required:
                # Read Data from training directory
                # Prepare Face data and prepare post directory data dir
                preproc.prepare_pre_procesing_folder_structure(self.configuration)
                preproc.prepare_faces_from_training_images_for_training(self.configuration)
                preproc.prepare_training_dictionary(self.configuration.post_processing_folder,self.configuration)
                preproc.prepare_training_testing_data(self.configuration)
                preproc.prepare_pickle_file(self.configuration)
            else:
                preproc.prepare_pre_procesing_folder_structure(self.configuration)
                preproc.prepare_training_dictionary(self.configuration.pre_processing_folder, self.configuration)
                preproc.prepare_training_testing_data(self.configuration)
                preproc.prepare_pickle_file(self.configuration)

            if configuration_details_json['classes'] == 'auto':
                number_of_classes = len(self.configuration.data.classes)
            else:
                number_of_classes = int(configuration_details_json['classes'])

            if configuration_details_json['network_config']['output_size'] == 'auto':
                self.configuration.network[-1]['weights'][1] = number_of_classes




