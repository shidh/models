import cv2
import os
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

PATH_TO_TEST_IMAGES_DIR = './test_images/'
class Detector(object):
    def __init__(self):
        self.PATH_TO_CKPT = r'../stamp_graph_all_stamps/frozen_inference_graph.pb'    # 选择模型
        self.PATH_TO_LABELS = r'./data/object-detection.pbtxt'      # 选择类别标签文件
        self.NUM_CLASSES = 9
        self.detection_graph = self._load_model()
        self.category_index = self._load_label_map()
    def _load_model(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph
    def _load_label_map(self):
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=self.NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index
    def detect(self, image_path):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                image = cv2.imread(image_path)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image, axis=0)
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    self.category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                #image = cv2.resize(image, (800, 600))

                # Classify the Positive/Negative results
                TEST_P_RESULT_DIR = "./test_result/P/"
                TEST_N_RESULT_DIR = "./test_result/N/"
                os.makedirs(TEST_P_RESULT_DIR, exist_ok=True)
                os.makedirs(TEST_N_RESULT_DIR, exist_ok=True)
                evaluate_file_path = TEST_P_RESULT_DIR + os.path.basename(image_path);
                if(scores[0][0] < 0.5):
                    evaluate_file_path = TEST_N_RESULT_DIR + os.path.basename(image_path);
                print(evaluate_file_path)
                cv2.imwrite(evaluate_file_path, image)
        #cv2.namedWindow("detection", cv2.WINDOW_NORMAL)
        #image = cv2.resize(image, (860, 730))
        #cv2.imshow("detection", image)
        #cv2.waitKey(0)
if __name__ == '__main__':
    imgs = [f for f in os.listdir(PATH_TO_TEST_IMAGES_DIR) if 'jpg' in f]
    for image_name in imgs:
        detector = Detector()
        detector.detect(PATH_TO_TEST_IMAGES_DIR + image_name)
