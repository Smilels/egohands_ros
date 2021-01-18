#!/usr/bin/env python3

# -- IMPORT --
import numpy as np
import cv2
import torch
# Ros
import rospy
import cv_bridge
import ros_numpy
import message_filters
from sensor_msgs.msg import Image, CompressedImage
# Ros egohands
from helper_CSAILVision.lib.segmentation import hand_segmentation, module_init
from IPython import embed

class Egohands:
    def __init__(self):
        # Parameter
        self.camera_topic = rospy.get_param('/egohands/camera/topic')
        self.depth_topic = rospy.get_param('/egohands/depth/topic')
        self.interface_topic = rospy.get_param('/egohands/interface/topic')
        self.visualization_topic = rospy.get_param('/egohands/visualization/topic')
        self.visualization_activated = rospy.get_param('/egohands/visualization/activated')
        print(self.interface_topic)

        # Init
        self.bridge = cv_bridge.CvBridge()
        self.segmentation_module = module_init()
        torch.cuda.set_device(0)
        
        # -- Visualization
        if self.visualization_activated:
            self.pub_visualization = rospy.Publisher(self.visualization_topic, Image, queue_size=1)

        # Subscriber
        image_sub = message_filters.Subscriber(self.camera_topic, CompressedImage)
        depth_sub = message_filters.Subscriber(self.depth_topic, Image)
        ts = message_filters.TimeSynchronizer([image_sub, depth_sub], 20)
        ts.registerCallback(self._callback)
        # Feedback
        print("Hand segmentation publisher up and running")

    # Callback function
    def _callback(self, rgb_img_data, depth_img_data):

        t_start = rospy.get_time()

        # Get image
        image = cv2.cvtColor(self.bridge.compressed_imgmsg_to_cv2(rgb_img_data), cv2.COLOR_BGR2RGB)
        depth = ros_numpy.numpify(depth_img_data).astype(np.float32)
     
        # Calculate Mask                 
        mask = hand_segmentation(image, self.segmentation_module)
        w,l = depth.shape

        # Visualize results
        if self.visualization_activated:
            image[:,:,0][mask == 0] = 0
            image[:,:,1][mask == 0] = 0
            image[:,:,2][mask == 0] = 0
            vis_img = ros_numpy.msgify(Image, image, encoding='8UC3')
            vis_img.header.frame_id = 'camera_color_optical_frame'
            # print(vis_img.header)
            self.pub_visualization.publish(vis_img)

        try:
            padding = 10
            box_z=250
            mask = mask.astype(np.bool)

            edge_x, edge_y = np.where(mask == 1)
            x_min, x_max = np.min(edge_x), np.max(edge_x)
            y_min, y_max = np.min(edge_y), np.max(edge_y)

            x_min = max(0, x_min - padding)
            x_max = min(x_max + padding, w - 1)
            y_min = max(0, y_min - padding)
            y_max = min(y_max + padding, l - 1)
            if x_max - x_min > y_max - y_min:
                delta = (x_max - x_min) - (y_max - y_min)
                y_min -= delta / 2
                y_max += delta / 2
            else:
                delta = (y_max - y_min) - (x_max - x_min)
                x_min -= delta / 2
                x_max += delta / 2
            x_min = int(max(0, x_min))
            x_max = int(min(x_max, w - 1))
            y_min = int(max(0, y_min))
            y_max = int(min(y_max, l - 1))

            np.nan_to_num(depth, copy=False)
            depth[np.where(depth>450)] = 0
            edge_depth = depth[np.where(mask == 1)]
            avg_depth = np.sum(edge_depth) / float(len(edge_depth))
            depth_min = max(avg_depth - box_z / 2, 0)
            depth_max = avg_depth + box_z / 2
            seg_area = depth.copy()
            seg_area[seg_area < depth_min] = depth_min
            seg_area[seg_area > depth_max] = depth_max
            # normalized
            seg_area = ((seg_area - avg_depth) / (box_z / 2))  # [-1, 1]
            seg_area = ((seg_area + 1) / 2.) * 255.  # [0, 255]

            output = seg_area[x_min:x_max, y_min:y_max]
            output = cv2.resize(output, (96, 96)).astype(np.uint16)
          
            output = output.astype(np.float32)
            output = output / 255. * 2. - 1
   
            n1 = cv2.normalize(output, output, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            cv2.imshow("segmented human hand", n1)
            cv2.waitKey(1) 
        except:
            pass

        # Publish results
        print('Hand detection successful. Current Hz-rate:\t' + str(1/(rospy.get_time() - t_start)))


if __name__ == '__main__':

    rospy.init_node('egohands_publisher')
    body = Egohands()
    rospy.spin()
