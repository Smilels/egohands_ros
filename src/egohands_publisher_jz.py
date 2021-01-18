#!/usr/bin/env python3

# -- IMPORT --
import numpy as np
import cv2
import torch
# Ros
import rospy
import cv_bridge
import ros_numpy
from sensor_msgs.msg import Image, CompressedImage, PointCloud2, PointField
from std_msgs.msg import Header
# Ros egohands
from helper_CSAILVision.lib.segmentation import hand_segmentation, module_init
import message_filters
# import tf2_ros
# import tf2_geometry_msgs
from scipy.spatial.transform import Rotation as R

show_3d = 1

# - Translation: [1.573, 2.066, 1.381]
# - Rotation: in Quaternion [0.124, 0.773, -0.614, -0.107]
#             in RPY (radian) [-1.797, -0.014, 2.813]
#             in RPY (degree) [-102.952, -0.810, 161.186]

class Egohands:
    def __init__(self):
        # Parameter
        self.camera_topic = rospy.get_param('/egohands/camera/topic')
        self.depth_topic = rospy.get_param('/egohands/depth/topic')
        self.interface_topic = rospy.get_param('/egohands/interface/topic')
        self.visualization_topic = rospy.get_param('/egohands/visualization/topic')
        self.visualization_activated = rospy.get_param('/egohands/visualization/activated')
        self.hand_pc_topic = "/HDgrasp_hand"
        self.hand_pc_pub = rospy.Publisher(self.hand_pc_topic, PointCloud2, queue_size=5)
        self.object_pc_topic = "/HDgrasp_object"
        self.object_pc_pub = rospy.Publisher(self.object_pc_topic, PointCloud2, queue_size=5)
        self.object_hand_pc_topic = "/HDgrasp_object_hand"
        self.object_hand_pc_pub = rospy.Publisher(self.object_hand_pc_topic, PointCloud2, queue_size=5)

        # Init
        self.bridge = cv_bridge.CvBridge()
        self.segmentation_module = module_init()
        torch.cuda.set_device(0)

        self.K = np.array([523.9900962215812, 0.0, 490.6032806218144,
                           0.0, 522.3086156124924, 278.5809640191108,
                           0.0, 0.0, 1.0]).reshape((3, 3))  ## need to be consistant to self.camera_topic

        ## for kinectic ##
        self.camera_to_world_rotation = R.from_quat([-0.833, 0.005, -0.006, 0.554]).as_matrix()
        self.camera_to_world_trans = np.array([0.620, -0.107, 0.480]).reshape((1, 3))
        ## for kinectic ##

        # Publisher
        # -- Mask
        self.pub_mask = rospy.Publisher(self.interface_topic, CompressedImage, queue_size=1)
        # self.r, self.t = self.transfor_to_world()
        self.t = np.array([1.573, 2.066, 1.381]).reshape((1, 3))
        self.t = self.t * 1000 # transfer to mm
        self.r = R.from_quat([0.124, 0.773, -0.614, -0.107]).as_matrix()

        # -- Visualization
        if self.visualization_activated:
            self.pub_visualization = rospy.Publisher(self.visualization_topic, Image, queue_size=5)

        image_sub = message_filters.Subscriber(self.camera_topic, CompressedImage, queue_size=10)
        depth_sub = message_filters.Subscriber(self.depth_topic, CompressedImage, queue_size=10)
        ts = message_filters.TimeSynchronizer([image_sub, depth_sub], 20)
        ts.registerCallback(self._callback)

        # imag = Image()
        # egohand_sub = rospy.Subscriber("/egohands/topic", Image, imag)

        # Feedback
        print("Hand segmentation publisher up and running")

    # Callback function
    def _callback(self, msg, depth_msg):
        global index_file
        t_start = rospy.get_time()

        # Get image
        image = cv2.cvtColor(self.bridge.compressed_imgmsg_to_cv2(msg), cv2.COLOR_BGR2RGB)

        ############# Added by Ge #####################
        image_color = np.reshape(image, [-1,3])
        ############# Added by Ge #####################

        # Calculate Mask
        mask = hand_segmentation(image, self.segmentation_module)
        bo = (mask == 1)
        bo1 = bo.reshape((-1,))

        np_arr = np.fromstring(depth_msg.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_ANYDEPTH)
        image_np = image_np.astype(np.float32, copy=True)

        # Removing points that are farther than 1 meter or missing depth values.
        np.nan_to_num(image_np, copy=False)
        mask_filter = np.where(np.logical_or(image_np == 0, image_np > 1000))
        image_np[mask_filter] = np.nan

        # filter and transform depth image to point cloud in the cage_table_top_link frame
        pc_all, pc_raw, selection = self.backproject(image_np, self.K, bo1, return_selection=True, return_finite_depth=True)
        pc_raw = pc_raw / 1000 # change point cloud into meter
        pc_all = pc_all / 1000 # change point cloud into meter
        pc_colors = image_color[selection] ## for mlab to plot
        pc_colors_rviz = pc_colors / 225 ## for rviz, if we do not devide 225, all point will be white

        if pc_colors.shape[0] > 10:
            ## for hand point clouds publish ##
            pc_colors_A = np.ones((pc_colors.shape[0], 1))
            pc_raw_vis = np.hstack((pc_raw, pc_colors_rviz, pc_colors_A))
            # print(pc_raw_vis.shape)
            # print(pc_raw_vis.max(axis=0))
            self.pc_msg = self.point_cloud(pc_raw_vis, "kinect2_link")
            self.hand_pc_pub.publish(self.pc_msg)
            ## for hand point clouds publish ##

        if pc_raw.shape[0] > 100:
            padding = 0.15
            scale1 = 3
            scale2 = 3
            scale3 = 3

            x_min_max = [np.min(pc_raw[:, 0]) - padding / scale1, np.max(pc_raw[:, 0]) + padding / scale1]
            y_min_max = [np.min(pc_raw[:, 1]) - padding / scale2, np.max(pc_raw[:, 1]) + padding / scale2]
            z_min_max = [np.min(pc_raw[:, 2]) - padding / scale3, np.max(pc_raw[:, 2]) + padding / scale3]

            selection_object_hand = np.where((pc_all[:, 0] > x_min_max[0]) & (pc_all[:, 0] < x_min_max[1])
                                                   & (pc_all[:, 1] > y_min_max[0]) & (pc_all[:, 1] < y_min_max[1])
                                                   & (pc_all[:, 2] > z_min_max[0]) & (pc_all[:, 2] < z_min_max[1]))

            points = pc_all[selection_object_hand]

            ############# Added by Ge #####################
            color = image_color[selection_object_hand]
            ############# Added by Ge #####################

            ## for hand and object point clouds publish ##
            color_A = np.ones((color.shape[0], 1))
            object_hand_pc_raw_vis = np.hstack((points, color / 255, color_A))
            self.object_hand_pc_msg = self.point_cloud(object_hand_pc_raw_vis, "kinect2_link")
            self.object_hand_pc_pub.publish(self.object_hand_pc_msg)
            ## for hand and object point clouds publish ##

            ## for object point clouds publish ##
            object_selection = np.where((pc_all[:, 0] > (x_min_max[0]) + 0.05) & (pc_all[:, 0] < x_min_max[1])
                                                   & (pc_all[:, 1] > y_min_max[0]) & (pc_all[:, 1] < y_min_max[1])
                                                   & (pc_all[:, 2] > z_min_max[0]) & (pc_all[:, 2] < z_min_max[1])
                                        & (bo1 == 0))
            object_points = pc_all[object_selection]
            object_color = image_color[object_selection]

            object_color_A = np.ones((object_color.shape[0], 1))
            object_pc_raw_vis = np.hstack((object_points, object_color / 255, object_color_A))
            self.object_pc_msg = self.point_cloud(object_pc_raw_vis, "kinect2_link")
            self.object_pc_pub.publish(self.object_pc_msg)
            ## for object point clouds publish ##

            # with open('/informatik2/tams/home/lyu/workspace/ws_hdover_melodic/src/egohands_ros/src/cloud_hand_object_' + str(index_file) + '.npy', "wb") as f:
            #     print(index_file)
            #     np.save(f, np.hstack((points, color)))
            # f.close()
            #
            # with open('/informatik2/tams/home/lyu/workspace/ws_hdover_melodic/src/egohands_ros/src/cloud_hand_only_' + str(index_file) + '.npy', "wb") as f1:
            #     print(index_file)
            #     np.save(f1, np.hstack((points_raw_transform_position_hand, points_raw_transform_position_hand_color)))
            # f1.close()

            if points.shape[0] < 300:
                print("hand points is %d, which is less than 300. Maybe it's a broken image" % (len(points)))
                return

            # Visualize results
            if self.visualization_activated:
                print("sss", image.shape)
                image[:, :, 0][mask == 0] = 0
                image[:, :, 1][mask == 0] = 0
                image[:, :, 2][mask == 0] = 0
                print(image.shape)
                self.pub_visualization.publish(ros_numpy.msgify(Image, image, encoding='8UC3'))

            # Publish results
            print('Hand detection successful. Current Hz-rate:\t' + str(1 / (rospy.get_time() - t_start)))
            # self.pub_mask.publish(self.bridge.cv2_to_compressed_imgmsg(mask, dst_format='png'))

    def point_cloud(self, points, parent_frame):
        """ Creates a point cloud message.
        Args:
            points: Nx7 array of xyz positions (m) and rgba colors (0..1)
            parent_frame: frame in which the point cloud is defined
        Returns:
            sensor_msgs/PointCloud2 message
        """
        ros_dtype = PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize

        data = points.astype(dtype).tobytes()

        fields = [PointField(
            name=n, offset=i * itemsize, datatype=ros_dtype, count=1)
            for i, n in enumerate('xyzrgba')]

        header = Header(frame_id=parent_frame, stamp=rospy.Time.now())

        return PointCloud2(
            header=header,
            height=1,
            width=points.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=(itemsize * 7),
            row_step=(itemsize * 7 * points.shape[0]),
            data=data
        )

    def backproject(self, depth_cv, intrinsic_matrix, bo1, return_finite_depth=True, return_selection=False):
        depth = depth_cv.astype(np.float32, copy=True)

        # get intrinsic matrix
        K = intrinsic_matrix
        Kinv = np.linalg.inv(K)

        # compute the 3D points
        width = depth.shape[1]
        height = depth.shape[0]

        # construct the 2D points matrix
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        ones = np.ones((height, width), dtype=np.float32)
        x2d = np.stack((x, y, ones), axis=2).reshape(width * height, 3)

        # backprojection
        R = np.dot(Kinv, x2d.transpose())

        # compute the 3D points
        X = np.multiply(np.tile(depth.reshape(1, width * height), (3, 1)), R)
        X = np.array(X).transpose()

        ## Take care the difference between
        ## X_in_world = np.dot(X, self.camera_to_world_rotation) + self.camera_to_world_trans * 1000 and next line
        X_in_world = np.dot(self.camera_to_world_rotation, X.transpose()).transpose() + self.camera_to_world_trans * 1000

        if return_finite_depth:
            # selection = np.where((np.isfinite(X[:, 0])) & (X_in_world[:, 2] > 10) & (bo1 == 1))
            selection = np.where((np.isfinite(X[:, 0])) & (X_in_world[:, 2] > 10) & (bo1 == 1))
            X_finite = X[selection]

        if return_selection:
            return X, X_finite, selection

        return X, X_finite

    # def transfor_to_world(self):
    #     tf_buffer = tf2_ros.Buffer(rospy.Duration(100))
    #     tf_listener = tf2_ros.TransformListener(tf_buffer)
    #     transform = tf_buffer.lookup_transform("/world", "/kinect2_link", rospy.Time())
    #     r = R.from_quat([transform.transform.rotation.x,
    #                      transform.transform.rotation.y,
    #                      transform.transform.rotation.z,
    #                      transform.transform.rotation.w])
    #
    #     return r.as_matrix(), np.array([transform.transform.translation.x,
    #                                     transform.transform.translation.y,
    #                                     transform.transform.translation.z]).reshape((1, 3))


if __name__ == '__main__':
    rospy.init_node('egohands_publisher')
    body = Egohands()
    rospy.spin()