import rosbag
import os, json
import numpy as np
from PIL import Image

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import ros_numpy
import pypcd4
from rospy_message_converter import message_converter
from pypcd4 import PointCloud
from sensor_msgs import point_cloud2


class GndRosbagProcess:
    def __init__(self, rosbag_path, bag_name, output_path, ref_name):
        self.rosbag_path = rosbag_path
        self.bag_name = bag_name
        self.output_path = output_path
        self.ref_name = ref_name

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def rosbag_process(self):
        bag = rosbag.Bag(self.rosbag_path)
        for topic, msg, time in bag.read_messages():
            if not os.path.exists(self.output_path + topic):
                os.makedirs(self.output_path + topic)

            if topic == "/camera_processed":
                info = self.img_processing(topic, msg, time)
            elif topic == "/ouster/points":
                info = self.point_cloud_processing(topic, msg, time)
            else:
                # info = self.message_processing(topic, msg, time)
                info = message_converter.convert_ros_message_to_dictionary(msg)
            self.json_store(topic, info, time)

    def json_store(self, topic, info, time):
        # Load the scene image path
        topic_ref_path = self.output_path + topic + "/" + self.ref_name
        if os.path.exists(topic_ref_path):
            with open(topic_ref_path, "r") as f:
                topic_ref = json.load(f)
        else:
            topic_ref = {}
            with open(topic_ref_path, 'w') as f:
                json.dump(topic_ref, f, indent=2)

        topic_ref[str(time)] = info

        with open(topic_ref_path, 'w') as f:
            json.dump(topic_ref, f, indent=2)

    def img_processing(self, topic, msg, time):
        im = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

        new_img = np.zeros_like(im)
        new_img[:, :, 0] = im[:, :, 2]
        new_img[:, :, 1] = im[:, :, 1]
        new_img[:, :, 2] = im[:, :, 0]

        img = Image.fromarray(new_img)
        img_path = self.output_path + topic + "/" + str(time) + ".png"
        img.save(img_path)
        
        return img_path

    # def message_processing(self, topic, msg, time):
    #     dictionary = message_converter.convert_ros_message_to_dictionary(msg)
    #     return dictionary

    def point_cloud_processing(self, topic, msg, time):
        gen = point_cloud2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
        pc_array = []

        for p in gen:
            pc_array.append(np.array(p))
        pc = PointCloud.from_xyzi_points(np.array(pc_array))

        pcd_path = self.output_path + topic + "/" + str(time) + ".pcd"
        pc.save(pcd_path)
        return pcd_path

rosbag_path = "/media/jim/Hard Disk/GND/UMD/map1/"
rosbag_name = "UMD_map1_1_trail_chunk01"
output_path = f"{rosbag_path}/{rosbag_name}"
ref_name = "info.json"

bag_pass = GndRosbagProcess(rosbag_path, rosbag_name, output_path, ref_name)
bag_pass.rosbag_process()