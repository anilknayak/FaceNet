import tensorflow as tf
from tensorflow.contrib.layers import flatten
from tqdm import tqdm
from tensorflow.python.platform import gfile

class Network:
    def __init__(self):
        self.sess = tf.InteractiveSession()
        self.input_tensor = None
        self.output_tensor = None
        self.output = None
        self.cross_entropy = None
        self.loss_operation = None
        self.optimizer = None
        self.model = None
        self.correct_prediction = None
        self.accuracy_operation = None
        self.output_tensor_one_hot = None
        self.learning_rate = 0.001
        self.training_epochs = 100
        self.batch_size = 24
        self.train = None
        self.test = None
        self.network = None
        self.class_number = None
        # self.saver = tf.train.Saver()

    def prepare(self,configuration):
        self.learning_rate = configuration.learning_rate
        self.training_epochs = configuration.training_steps
        self.batch_size = configuration.batch_size
        self.train = configuration.data.train_data
        self.test = configuration.data.test_data
        self.network = configuration.network
        self.class_number = configuration.data.classes_count

    def conv_layer(self, prev_layer, layer):
        W = tf.Variable(tf.random_normal(layer['weights'], dtype=tf.float32), dtype=tf.float32)
        B = tf.Variable(tf.random_normal([layer['weights'][-1]], dtype=tf.float32), dtype=tf.float32)
        convolution = tf.nn.conv2d(prev_layer, W, strides=layer['strides'], padding=layer['padding'],
                                   name=layer['name']) + B
        return convolution

    def maxpool_layer(self, prev_layer, layer):
        max_pooling = tf.nn.max_pool(prev_layer, ksize=layer['filters'], strides=layer['strides'],
                                     padding=layer['padding'], name=layer['name'])
        return max_pooling

    def relu_layer(self, prev_layer, layer):
        if layer['flatten']:
            return flatten(tf.nn.relu(prev_layer, name='flatten_' + layer['name']))
        else:
            return tf.nn.relu(prev_layer, name=layer['name'])

    def dense_layer(self, prev_layer, layer):
        fw = tf.Variable(tf.random_normal(layer['weights'], dtype=tf.float32), dtype=tf.float32)
        fb = tf.Variable(tf.random_normal([layer['weights'][-1]], dtype=tf.float32), dtype=tf.float32)
        fc = tf.add(tf.matmul(prev_layer, fw), fb)
        return fc

    def build_model(self):
        classes_number = len(self.class_number)
        neural_network_dict = self.network
        layers_op = []
        self.input_tensor = tf.placeholder(tf.float32,
                                           [None, neural_network_dict[0]['width'], neural_network_dict[0]['height'], 1],
                                           name='input_tensor')
        self.output_tensor = tf.placeholder(tf.int32, (None))
        self.output_tensor_one_hot = tf.one_hot(self.output_tensor, classes_number)

        layers_op.append(self.input_tensor)

        for layer in neural_network_dict:
            if layer['type'] == 'conv':
                layers_op.append(self.conv_layer(layers_op[-1], layer))
            elif layer['type'] == 'maxpool':
                layers_op.append(self.maxpool_layer(layers_op[-1], layer))
            elif layer['type'] == 'relu':
                layers_op.append(self.relu_layer(layers_op[-1], layer))
            elif layer['type'] == 'fc':
                layers_op.append(self.dense_layer(layers_op[-1], layer))

        self.output = layers_op[-1]
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.output_tensor_one_hot)
        self.loss_operation = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.model = self.optimizer.minimize(self.loss_operation)
        self.correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.output_tensor_one_hot, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def train_model(self):
        images = self.train['images']
        labels = self.train['labels_n']

        saver = tf.train.Saver()
        with self.sess.as_default():
            self.sess.run(tf.global_variables_initializer())

            for i in range(self.training_epochs):
                epoch_loss = 0
                start = 0
                end = 0
                for j in tqdm(range(0, len(images), self.batch_size)):
                    end = start + self.batch_size
                    batch_images = images[start:end]
                    # batch_labels = images[start:end]
                    batch_labels = labels[start:end]
                    start = start + self.batch_size

                    _, loss = self.sess.run([self.model, self.loss_operation],
                                       feed_dict={self.input_tensor: batch_images,
                                                  self.output_tensor: batch_labels})

                    epoch_loss += loss

                print("Epoch Loss for epoch : ", str(i), " is ", epoch_loss)

            graph = tf.get_default_graph()
            saver.save(self.sess, './model/')
            tf.train.write_graph(graph, './model/', 'train.pb')

    def test_model(self):
        images = self.test['images']
        labels = self.test['labels_n']

        with self.sess.as_default():
            print('Accuracy is ', self.accuracy_operation.eval({self.input_tensor: images, self.output_tensor: labels}))

    def load_pre_train_model(self,model_file):
        with gfile.FastGFile(model_file, "rb") as f:
            graph_def = tf.GraphDef()
            byte = f.read()
            graph_def.ParseFromString(byte)

        tf.import_graph_def(graph_def, name='')
