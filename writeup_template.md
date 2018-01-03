[image1]: ./confusion_matrix.png
[image2]: ./stats_filter.png
[image3]: ./vox.png
[image4]: ./passthrough.png
[image5]: ./cluster.png
[image6]: ./table.png
[image7]: ./objects.png
[image8]: ./camera.png
[image9]: ./camera_cropped.png


## Project: Perception Pick & Place
### Writeup Template: You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---


# Required Steps for a Passing Submission:
1. Extract features and train an SVM model on new objects (see `pick_list_*.yaml` in `/pr2_robot/config/` for the list of models you'll be trying to identify). 
2. Write a ROS node and subscribe to `/pr2/world/points` topic. This topic contains noisy point cloud data that you must work with.
3. Use filtering and RANSAC plane fitting to isolate the objects of interest from the rest of the scene.
4. Apply Euclidean clustering to create separate clusters for individual items.
5. Perform object recognition on these objects and assign them labels (markers in RViz).
6. Calculate the centroid (average in x, y and z) of the set of points belonging to that each object.
7. Create ROS messages containing the details of each object (name, pick_pose, etc.) and write these messages out to `.yaml` files, one for each of the 3 scenarios (`test1-3.world` in `/pr2_robot/worlds/`).  [See the example `output.yaml` for details on what the output should look like.](https://github.com/udacity/RoboND-Perception-Project/blob/master/pr2_robot/config/output.yaml)  
8. Submit a link to your GitHub repo for the project or the Python code for your perception pipeline and your output `.yaml` files (3 `.yaml` files, one for each test world).  You must have correctly identified 100% of objects from `pick_list_1.yaml` for `test1.world`, 80% of items from `pick_list_2.yaml` for `test2.world` and 75% of items from `pick_list_3.yaml` in `test3.world`.
9. Congratulations!  Your Done!

# Extra Challenges: Complete the Pick & Place
7. To create a collision map, publish a point cloud to the `/pr2/3d_map/points` topic and make sure you change the `point_cloud_topic` to `/pr2/3d_map/points` in `sensors.yaml` in the `/pr2_robot/config/` directory. This topic is read by Moveit!, which uses this point cloud input to generate a collision map, allowing the robot to plan its trajectory.  Keep in mind that later when you go to pick up an object, you must first remove it from this point cloud so it is removed from the collision map!
8. Rotate the robot to generate collision map of table sides. This can be accomplished by publishing joint angle value(in radians) to `/pr2/world_joint_controller/command`
9. Rotate the robot back to its original state.
10. Create a ROS Client for the “pick_place_routine” rosservice.  In the required steps above, you already created the messages you need to use this service. Checkout the [PickPlace.srv](https://github.com/udacity/RoboND-Perception-Project/tree/master/pr2_robot/srv) file to find out what arguments you must pass to this service.
11. If everything was done correctly, when you pass the appropriate messages to the `pick_place_routine` service, the selected arm will perform pick and place operation and display trajectory in the RViz window
12. Place all the objects from your pick list in their respective dropoff box and you have completed the challenge!
13. Looking for a bigger challenge?  Load up the `challenge.world` scenario and see if you can get your perception pipeline working there!

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

In order to correctly identify the objects on the top of the table, so the robot could pick and place them, a perception pipeline was to be created that consists of Filtering, Segmentation, Clustering, and a simple supervised learning algorithm to train a model to recognize the different objects based on their features which comprise of their respective color histogram, in addition to their surface normal histogram.
Prior to the implementation of the project, three exercises provided in the lectures introduced insight to the most of those different aspects of the pipleline, and were implemented incrementally to finally end up with a script that could identify the objects and publish their labels in RViz.
In addition to the above mentioned steps, the main project required the population of a function called `pr2_mover()` that would be called after perception pipleline, that would send a service request populated with computed object location, which of the robot's arms would be used, in which bin the object should be dropped, among other information, in order for the robot to pick them up and place them in their respective dropboxes. Each of the steps is discussed in-depth in the following questions.

### Exercise 1, 2 and 3 pipeline implemented
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.

**1. Statistical Outlier Filter:** Since most sensor/camera readings are noisy, we first implement a __Statistial Outlier Filter__ to remove the majority of the noise from the readings. 

```python
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
```

![stats_filter][image2]

**2. Voxel Grid Downsampling:** Next, a __Voxel Grid__ filter is applied to the frame with reduced noise, to decrease the amount of points to process, since the redundancy of points/data does not contribute to the recognition process, since we can downsample and still keep the most important of features.

```python
    ###################################################
    # TODO: Voxel Grid Downsampling
    ###################################################
    
    vox = cloud_pcl.make_voxel_grid_filter()
    
    LEAF_SIZE = 0.006 # 0.005
    
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    
    voxed = vox.filter()
```

![vox][image3]

**3. Passthough Filter:** Such filters are used to basically __crop__ the scene and completely remove insignificant data. In our case we apply this type of filter twice, one in the `z` axis and one in the `x` axis.

```python
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
```


![passthrough][image4]

**4. RANSAC Plane Segmentation:** This segmentation algorithm finds points in the cloud that satisfy specific mathematical relations, in this case, points that belong to a plane.
__Once a model is established, the remaining points in the point cloud are tested against the resulting candidate shape to determine how many of the points are well approximated by the model. After a certain number of iterations, the shape that possesses the largest percentage of inliers is extracted and the algorithm continues to process the remaining data.__

```python
    ###################################################
    # TODO: RANSAC Plane Segmentation
    ###################################################
    seg = passthroughed.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()
```
The points are then extracted from the filtered frames, which correspond to the objects and table.

```python
    ###################################################
    # TODO: Extract inliers and outliers
    ###################################################
    extracted_inliers = passthroughed.extract(inliers, negative = False)
    extracted_outliers = passthroughed.extract(inliers, negative = True)
```


![table][image6]

![objects][image7]

#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  

Clustering is used to, as the name suggests, __cluster__ data points that are within a certain distance of each other, to define points belonging to an object. The code snippet is taken from the lessons, along with the visualization part. Several techniques like **DBSCAN** and **K-MEANS** searching algorithms were introduced in the lessons, and we stuck with the **K-MEANS** technique.

```python
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
```



![cluster][image5]

#### 3. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.

In this scenario, we know what the objects we are going to find are, and so, we can start extracting their features. We choose to extract __color histogram__ features along with __surface normals__ features, as those contain description about the object's color and one of its geometrical features, using `capture_features.py` script, which creates every object specified in the `object_list` a specified number of times in randem poses in front of the RGB-D camera. In my case, i chose the number of random orientations to be 25 so that i make sure that each object is captured from as many angles as possible so as to increase the robutness of the detection later on.
After we capture the features we mentioned above, we use a supervised learning algorthim (Support Vector Machines) to train a classifer to recognize the objects based on their features and their respective labels. I changed the default kernel to `sigmoid` and increased the cross-validation to `8` folds instead of 5, which resulted in very robust results with accuracy reaching `97%`.

```python
# Create classifier
clf = svm.SVC(kernel = 'sigmoid')

# Set up 5-fold cross-validation
kf = cross_validation.KFold(len(X_train),
                            n_folds=8, # 5
                            shuffle=True,
                            random_state=1)
```


![confusion_matrix][image1]


Now that we have isolated our objects of interest from the rest of the scene/data using the above mentioned pipeline, we can extract the features of those clusters, load our saved model, and perform classification.

```python
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
```

We then publish these detected objects for Rviz to display above each respective object. In addition, we pass the list of detected objects to the `pr2_mover()` function as an argument.

Camera Perspective:


![camera_cropped][image9]

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

Classifier accuracy increase was achieved by:
- Increasing number of random poses captured and their respective features (More data!)
- Changing classifier kernel to `sigmoid`

- Filter parameter tuning should be adjusted a little bit more for robustness

- Collision avoidance should be implemented in order to avoid hitting other objects while picking

- Gripping time should be increased (or strengthened?) since most objects slip through



