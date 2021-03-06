#!/usr/bin/env python3

# -- IMPORT --
import numpy as np
import cv2
import torch
# Ros
import rospy
import cv_bridge
import ros_numpy
import actionlib
from sensor_msgs.msg import Image, CompressedImage
# Ros egohands
from egohands_ros.msg import SemSegHandActAction, SemSegHandActResult
from helper_CSAILVision.lib.segmentation import hand_segmentation, module_init


class Egohands:
    def __init__(self):
        # Parameter
        self.camera_topic = rospy.get_param('/egohands/camera/topic')
        self.interface_topic = rospy.get_param('/egohands/interface/action')
        self.visualization_topic = rospy.get_param('/egohands/visualization/topic')
        self.visualization_activated = rospy.get_param('/egohands/visualization/activated')

        # Init
        self.bridge = cv_bridge.CvBridge()
        self.segmentation_module = module_init()
        torch.cuda.set_device(0)

        # Action server
        # -- Mask
        self.server = actionlib.SimpleActionServer(self.interface_topic, SemSegHandActAction, self._callback, False)
        self.server.start()

        # -- Visualization
        if self.visualization_activated:
            self.pub_visualization = rospy.Publisher(self.visualization_topic, Image, queue_size=1)

        # Feedback
        print("Hand segmentation action server up and running")

    # Callback function
    def _callback(self, goal):

        t_start = rospy.get_time()

        # Get image
        image = cv2.cvtColor(self.bridge.compressed_imgmsg_to_cv2(goal.image), cv2.COLOR_BGR2RGB)

        # Calculate Mask                 
        mask = hand_segmentation(image, self.segmentation_module)

        # Visualize results
        if self.visualization_activated:
            image[:,:,0][mask == 0] = 0
            image[:,:,1][mask == 0] = 0
            image[:,:,2][mask == 0] = 0
            self.pub_visualization.publish(ros_numpy.msgify(Image, image, encoding='8UC3'))
        
        # Publish results
        print('Body detection successful. Current Hz-rate:\t' + str(1/(rospy.get_time() - t_start)))
        self.server.set_succeeded(SemSegHandActResult(mask=self.bridge.cv2_to_compressed_imgmsg(mask, dst_format='png')))


if __name__ == '__main__':

    rospy.init_node('egohands_action_server')
    body = Egohands()    
    rospy.spin()