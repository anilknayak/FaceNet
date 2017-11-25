

class Configuration:
    def __init__(self):
        self.base_directory = None
        self.image_directory = None
        self.network = None
        self.image_h = None
        self.image_w = None
        self.image_resize_h = None
        self.image_resize_w = None

        self.learning_rate = None
        self.training_folder = None
        self.testing_folder = None
        self.pre_processing_folder = None
        self.post_processing_folder = None

        self.random_shuffle = None
        self.training_size_percentage = None
        self.training_steps = None
        self.batch_size = None
        self.pre_processing_required = None

        self.data = None

        self.pickle_data_file = None
        self.prepare_pickle_file = None

