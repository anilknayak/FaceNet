import setup


# neural_network_dict = [
#         {'name': 'conv1', 'type': 'conv', 'filters': [1, 5, 5, 1], 'strides': [1, 1, 1, 1], 'padding': 'SAME', 'width': width, 'height': height, 'weights': [5, 5, 1, 32]},
#         {'name': 'maxpool1', 'type': 'maxpool', 'filters': [1, 2, 2, 1], 'strides': [1, 2, 2, 1], 'padding': 'SAME'},
#         {'name': 'relu1', 'type': 'relu', 'flatten': False},
#         {'name': 'conv2', 'type': 'conv', 'filters': [1, 5, 5, 1], 'strides': [1, 1, 1, 1], 'padding': 'SAME', 'weights': [5, 5, 32, 64]},
#         {'name': 'maxpool2', 'type': 'maxpool', 'filters': [1, 2, 2, 1], 'strides': [1, 2, 2, 1], 'padding': 'SAME'},
#         {'name': 'relu2', 'type': 'relu', 'flatten': True},
#         {'name': 'fc1', 'type': 'fc', 'weights': [45 * 45 * 64, 512]},
#         {'name': 'relu3', 'type': 'relu', 'flatten': False},
#         {'name': 'output', 'type': 'fc', 'weights': [512, classes_number]}
#     ]

facenet = setup.SetUp()

print('Preparing Network')
facenet.network.prepare(facenet.configuration)
print('Building Network')
facenet.network.build_model()
print('Training Starts')
facenet.network.train_model()
print('Testing Starts')
facenet.network.test_model()








