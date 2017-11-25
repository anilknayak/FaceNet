import tensorflow as tf
import cv2
from tensorflow.python.platform import gfile

video_capture = cv2.VideoCapture(0)

frozen_graph_filename = 'model/train_model.pb'

with gfile.FastGFile(frozen_graph_filename, "rb") as f:
    graph_def = tf.GraphDef()
    byte = f.read()
    graph_def.ParseFromString(byte)

tf.import_graph_def(graph_def, name='')

for node in graph_def.node:
    print(node.name)

with tf.Session() as sess:
    detection_graph = tf.get_default_graph()
    input_tensor = detection_graph.get_tensor_by_name('input_tensor:0')
    output_tensor = detection_graph.get_tensor_by_name('output_tensor:0')
    output = detection_graph.get_tensor_by_name('output:0')
    result = detection_graph.get_tensor_by_name('result:0')

    while True:
        ret, frame = video_capture.read()
        gray_scale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_between_0_and_1 = gray_scale_image / 255.0
        image_between_0_and_1 = image_between_0_and_1 - 0.5
        normalized_image_between_ng_1_and_po_1 = image_between_0_and_1 * 2.0
        frame2 = normalized_image_between_ng_1_and_po_1[150:330, 300:480]
        frame1 = frame2.reshape((1, 180, 180, 1))
        (result1, output1) = sess.run([result, output], feed_dict={input_tensor: frame1, output_tensor: 1})

        print(result1, output1)

        cv2.imshow('Face Detection [press q to quit window]', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()