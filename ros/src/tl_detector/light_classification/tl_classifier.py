import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import rospy

from collections import defaultdict
from io import StringIO

from PIL import Image

from .object_detection.utils import label_map_util
from .object_detection.utils import visualization_utils as vis_util

from styx_msgs.msg import TrafficLight
from std_msgs.msg import String
from scipy.stats import mode

MODEL_NAME = 'light_classification/' + 'ssd_custom_graph'
print("MODEL_NAME = %s" % MODEL_NAME)
TF_VERSION = "1.3" # use 1.15 in Docker container
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph_tf_%s.pb' % TF_VERSION
PATH_TO_LABELS = 'light_classification/training/label_map.pbtxt'
NUM_CLASSES = 3 
SCORE_THRESH = 0.85
class_lookup = {
        1 : TrafficLight.GREEN,
        2 : TrafficLight.YELLOW,
        3 : TrafficLight.RED,
}

class NullContextManager(object):
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource
    def __enter__(self):
        return self.dummy_resource
    def __exit__(self, *args):
        pass

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.detection_graph, self.label_map, self.categories, self.category_index = self.import_graph()
        self.tf_config = tf.ConfigProto()
        self.tf_config.gpu_options.allow_growth = True
        # TODO: check if we need detection_graph.as_default here
        self.sess = tf.Session(graph=self.detection_graph, config=self.tf_config)
        # Run fake data during init to warm up TensorFlow's memory allocator
        warmup_iter = 10
        for iter in range(warmup_iter):
            synth_data = np.random.randint(low=0, high=255, size=(600, 800, 3), dtype=np.uint8)
            self.inference(synth_data)
        light_detector_pub = rospy.Publisher('/tl_detections', String, queue_size=1)
        light_detector_pub.publish(String("Light detector bootstrap executed. Synthetic data passed through model without errors."))

    def import_graph(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return detection_graph, label_map, categories, category_index

    def inference(self, image):
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        image_np_expanded = np.expand_dims(image, axis=0)

        (boxes, scores, classes, num) = self.sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        return boxes, scores, classes, num

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        boxes, scores, classes, num = self.inference(image)
        scores = scores[0]
        classes = classes[0]
        good_scores = np.argwhere(scores > SCORE_THRESH)
        good_classes = classes[good_scores]
        if len(good_scores) < 1:
            # No detections
            return TrafficLight.UNKNOWN
        class_mode = int(mode(good_classes)[0][0][0])
        return class_lookup[class_mode]
