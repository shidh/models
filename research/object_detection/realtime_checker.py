import cv2
import os
import datetime
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

cap = cv2.VideoCapture(0)

class Detector(object):
    def __init__(self):
        self.PATH_TO_CKPT = r'./stamp_graph/frozen_inference_graph.pb'
        self.PATH_TO_LABELS = r'./data/object-detection.pbtxt'
        self.NUM_CLASSES = 2
        '''
        self.PATH_TO_CKPT = r'./ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb'
        self.PATH_TO_LABELS = r'./data/mscoco_label_map.pbtxt'
        self.NUM_CLASSES = 400
        '''
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

    def detect(self, image):
        config = tf.ConfigProto(device_count={"CPU": 4}, # limit to num_cpu_core CPU usage
                inter_op_parallelism_threads = 4,
                intra_op_parallelism_threads = 4,
                log_device_placement=False)
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph, config=config) as sess:
                print("now =", datetime.datetime.now())
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image = cv2.resize(image, (400, 300))
                image_np_expanded = np.expand_dims(image, axis=0)
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                print("start =", datetime.datetime.now())
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                print("detect =", datetime.datetime.now())
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    self.category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
        cv2.namedWindow("detection", cv2.WINDOW_NORMAL)
        print("visualize =", datetime.datetime.now())
        # Resize image
        #image = cv2.resize(image, (800, 600))
        cv2.imshow("detection", image)
        #cv2.waitKey(0)
        print("end =", datetime.datetime.now())


if __name__ == '__main__':
    while True:
        ret, frame = cap.read()
        detector = Detector()
        detector.detect(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destoryAllWindows()
    #PATH_TO_TEST_IMAGES_DIR = 'test_images'
    #TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 21) ]
    #for image_path in TEST_IMAGE_PATHS:
    #    image = cv2.imread(image_path) # 选择待检测的图片
    #    detector = Detector()
    #    detector.detect(image)
