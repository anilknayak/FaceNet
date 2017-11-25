import tensorflow as tf
import cv2
from tensorflow.python.platform import gfile
import dlib
from imutils import face_utils

frozen_graph_filename = 'model/train_model.pb'

with gfile.FastGFile(frozen_graph_filename, "rb") as f:
    graph_def = tf.GraphDef()
    byte = f.read()
    graph_def.ParseFromString(byte)

tf.import_graph_def(graph_def, name='')

for node in graph_def.node:
    print(node.name)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("dlib_pretrained_model.dat")
image_path = "/test_image/image1.jpg"

with tf.Session() as sess:
    detection_graph = tf.get_default_graph()
    input_tensor = detection_graph.get_tensor_by_name('input_tensor:0')
    output_tensor = detection_graph.get_tensor_by_name('output_tensor:0')
    output = detection_graph.get_tensor_by_name('output:0')
    result = detection_graph.get_tensor_by_name('result:0')

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(image, 1)
    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face = image[y - 50: y + h + 10, x - 10: x + w + 20]
        face = cv2.resize(face, (180, 180), interpolation=cv2.INTER_CUBIC)


    gray_scale_image = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    image_between_0_and_1 = gray_scale_image / 255.0
    image_between_0_and_1 = image_between_0_and_1 - 0.5
    normalized_image_between_ng_1_and_po_1 = image_between_0_and_1 * 2.0
    print(normalized_image_between_ng_1_and_po_1.shape)
    frame1 = normalized_image_between_ng_1_and_po_1.reshape((1, 180, 180, 1))

    (result1, output1) = sess.run([result, output], feed_dict={input_tensor: frame1, output_tensor: 1})

    print(result1, output1)


