#!/usr/bin/env python

import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder

import pickle

from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker

from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster
    


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

        # TODO: Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    # TODO: Voxel Grid Downsampling
    vox = cloud.make_voxel_grid_filter()
    LEAF_SIZE = 0.005
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    voxed = vox.filter()

    # TODO: PassThrough Filter "Z"
    passthrough = voxed.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.5
    axis_max = 1.1
    passthrough.set_filter_limits(axis_min, axis_max)
    passthroughed_1 = passthrough.filter()

    # TODO: PassThrough Filter "Y"
    passthrough_2 = passthroughed_1.make_passthrough_filter()
    filter_axis_2 = 'y'
    passthrough_2.set_filter_field_name(filter_axis_2)
    axis_min_2 = -3
    axis_max_2 = -1.35
    passthrough_2.set_filter_limits(axis_min_2, axis_max_2)
    passthroughed = passthrough_2.filter()

    # TODO: RANSAC Plane Segmentation
    seg = passthroughed.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()

    # TODO: Extract inliers and outliers
    extracted_inliers = passthroughed.extract(inliers, negative = False)
    extracted_outliers = passthroughed.extract(inliers, negative = True)

    # TODO: Euclidean Clustering
    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(extracted_outliers)# Apply function to convert XYZRGB to XYZ
    tree = white_cloud.make_kdtree()



    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold 
    # as well as minimum and maximum cluster size (in points)
    # NOTE: These are poor choices of clustering parameters
    # Your task is to experiment and find values that work for segmenting objects.
    ec.set_ClusterTolerance(0.01)
    ec.set_MinClusterSize(0.5)
    ec.set_MaxClusterSize(10000)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()


    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    #Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # TODO: Convert PCL data to ROS messages
    cloud_table = pcl_to_ros(extracted_inliers)
    cloud_objects = pcl_to_ros(extracted_outliers)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # TODO: Publish ROS messages
    pcl_table_pub.publish(cloud_table)
    pcl_objects_pub.publish(cloud_objects)
    pcl_cluster_pub.publish(ros_cluster_cloud)

# Exercise-3 TODOs: 

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []
    
    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = extracted_outliers.extract(pts_list)
        ros_cluster = pcl_to_ros(pcl_cluster)
        
        # Compute the associated feature vector
        hsv_hist = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        normals_hist = compute_normal_histograms(normals)
        feature = np.concatenate((hsv_hist, normals_hist))
 
        
        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    # This is the output you'll need to complete the upcoming project!
    detected_objects_pub.publish(detected_objects)

if __name__ == '__main__':

# TODO: ROS node initialization
    rospy.init_node('clustering', anonymous = True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber('/sensor_stick/point_cloud', pc2.PointCloud2, pcl_callback, queue_size = 1)
    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher('/pcl_objects', PointCloud2, queue_size = 1)
    pcl_table_pub = rospy.Publisher('/pcl_table', PointCloud2, queue_size = 1)
    pcl_cluster_pub = rospy.Publisher('/pcl_cluster', PointCloud2, queue_size = 1)
    object_markers_pub = rospy.Publisher('/object_markers', Marker, queue_size = 1)
    detected_objects_pub = rospy.Publisher('/detected_objects', DetectedObject, queue_size = 1)
    # TODO: Load Model From disk
    model = pickle.load(open('/home/robond/catkin_ws/model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']
    
    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
