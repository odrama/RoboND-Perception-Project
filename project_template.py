#!/usr/bin/env python

# Import modules
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

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    ###################################################
    # TODO: Convert ROS msg to PCL data
    ###################################################
    cloud_pcl = ros_to_pcl(pcl_msg)
    
    
    ###################################################
    # TODO: Statistical Outlier Filtering
    ###################################################
    # Much like the previous filters, we start by creating a filter object: 
    outlier_filter = cloud_pcl.make_statistical_outlier_filter()

    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(100)

    # Set threshold scale factor
    x = 0.0006

    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)

    # Finally call the filter function for magic
    pcl_cloud_filtered = outlier_filter.filter()
    
    # test_msg = pcl_to_ros(pcl_cloud_filtered)
    # pcl_test_pub.publish(test_msg)
    
    ###################################################
    # TODO: Voxel Grid Downsampling
    ###################################################
    
    vox = cloud_pcl.make_voxel_grid_filter()
    
    LEAF_SIZE = 0.006 # 0.005
    
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    
    voxed = vox.filter()
    
    # test_msg = pcl_to_ros(voxed)
    # pcl_test_pub.publish(test_msg)
    
    ###################################################
    # TODO: PassThrough Filter
    ###################################################
    passthrough = voxed.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.5
    axis_max = 1.1
    passthrough.set_filter_limits(axis_min, axis_max)
    passthroughed_1 = passthrough.filter()

    # TODO: PassThrough Filter "X"
    passthrough_2 = passthroughed_1.make_passthrough_filter()
    filter_axis_2 = 'x'
    passthrough_2.set_filter_field_name(filter_axis_2)
    axis_min_2 = 0.4
    axis_max_2 = 0.8
    passthrough_2.set_filter_limits(axis_min_2, axis_max_2)
    passthroughed = passthrough_2.filter()
    
    # test_msg = pcl_to_ros(passthroughed)
    # pcl_test_pub.publish(test_msg)
    
    ###################################################
    # TODO: RANSAC Plane Segmentation
    ###################################################
    seg = passthroughed.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()
    
    ###################################################
    # TODO: Extract inliers and outliers
    ###################################################
    extracted_inliers = passthroughed.extract(inliers, negative = False)
    extracted_outliers = passthroughed.extract(inliers, negative = True)
    
    
    cloud_table = pcl_to_ros(extracted_inliers)
    cloud_objects = pcl_to_ros(extracted_outliers)
        
    pcl_table_pub.publish(cloud_table)
    pcl_objects_pub.publish(cloud_objects)

    ###################################################
    # TODO: Euclidean Clustering
    ###################################################
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
    ec.set_MinClusterSize(5) # 0.5
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
    
    
    
    
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)
    pcl_cluster_pub.publish(ros_cluster_cloud)
    

    ###################################################
    # TODO: Convert PCL data to ROS messages
    ###################################################
    
    ###################################################
    # TODO: Publish ROS messages
    ###################################################

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

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    detected_objects_list = detected_objects
    
    
    if len(detected_objects_list) >= len(rospy.get_param('/object_list')):
        print('Objects correctly identified!')
    
        try:
            pr2_mover(detected_objects_list)
        except rospy.ROSInterruptException:
            pass
    else:
        print('Number of detected objects is not equal to number of objects in the scene!!!')

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables
    labels = []
    centroids = []
    dict_list = []
    test_scene_num = Int32()
    object_name = String()
    arm_name = String()
    pick_pose = Pose()
    place_pose = Pose()
    test_scene_num.data = 3         # Test World, Dont forget to change!!!!
    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')  # Provided
    
    dropbox_param = rospy.get_param('/dropbox')
    for i in range(len(dropbox_param)):
        dropbox_name = dropbox_param[i]['name']
        if dropbox_name == 'left':
            dropbox_left_position = dropbox_param[i]['position']
        else:
            dropbox_right_position = dropbox_param[i]['position']
	
    # TODO: Parse parameters into individual variables
    
    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list
    """
    for idx, object in enumerate(object_list):
        
        object_name = object_list_param[idx]['name']      # Provided
        object_group = object_list_param[idx]['group']    # Provided
        object_name.data = object_name
        if object_group == 'green':
            arm_name.data = 'right'
            place_pose.data = dropbox_right_position
        elif object_group == 'red':
            arm_name.data = 'left'
            place_pose.data = dropbox_left_position
            
        
        # TODO: Get the PointCloud for a given object and obtain it's centroid
        labels.append(object.label)
        points_arr = ros_to_pcl(object.cloud).to_array()
        centroids.append(np.asscalar(np.mean(points_arr, axis=0)[:3]))
        pick_pose.data = centroids[-1]
        """
    for object in object_list:
        labels.append(object.label)
        points_arr = ros_to_pcl(object.cloud).to_array()
        avg = np.mean(points_arr, axis=0)[:3]
        x = np.asscalar(avg[0])
        y = np.asscalar(avg[1])
        z = np.asscalar(avg[2])
        centroid_asscalar = (x, y, z)
        centroids.append(centroid_asscalar)
        
        # centroids.append(np.asscalar(np.mean(points_arr, axis=0)[:3]))
        # pick_pose.data = centroids[-1]    
    
    
    for idx in range(len(object_list_param)):
    
        name = object_list_param[idx]['name']      # Provided
        group = object_list_param[idx]['group']    # Provided
        object_name.data = name
        
        if group == 'green':
            arm_name.data = 'right'
            # place_pose.data = dropbox_right_position
            place_pose.position.x = dropbox_right_position[0]
            place_pose.position.y = dropbox_right_position[1]
            place_pose.position.z = dropbox_right_position[2]
        elif group == 'red':
            arm_name.data = 'left'
            # place_pose.data = dropbox_left_position
            place_pose.position.x = dropbox_left_position[0]
            place_pose.position.y = dropbox_left_position[1]
            place_pose.position.z = dropbox_left_position[2]
            
        if name in labels:
            # pick_pose.data = centroids[labels.index[name]]
            print(name)
            pick_pose.position.x = centroids[labels.index(name)][0]
            pick_pose.position.y = centroids[labels.index(name)][1]
            pick_pose.position.z = centroids[labels.index(name)][2]
            
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        dict_list.append(yaml_dict)
        
        # TODO: Create 'place_pose' for the object

        # TODO: Assign the arm to be used for pick_place

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
    
    
        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
            resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file
    send_to_yaml('output_3.yaml', dict_list)


if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('project_node', anonymous = True)

    # TODO: Create Subscribers
    rospy.Subscriber('/pr2/world/points', pc2.PointCloud2, pcl_callback, queue_size = 1)
	
    # TODO: Create Publishers
    pcl_test_pub = rospy.Publisher('/pcl_test', PointCloud2, queue_size = 1)
    pcl_table_pub = rospy.Publisher('/pcl_table', PointCloud2, queue_size = 1)
    pcl_objects_pub = rospy.Publisher('/pcl_objects', PointCloud2, queue_size = 1)
    pcl_cluster_pub = rospy.Publisher('/pcl_cluster', PointCloud2, queue_size = 1)
    object_markers_pub = rospy.Publisher('/object_markers', Marker, queue_size = 1)
    detected_objects_pub = rospy.Publisher('/detected_objects', DetectedObjectsArray, queue_size = 1)
    
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
