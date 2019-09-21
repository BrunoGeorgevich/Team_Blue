#!/usr/bin/env python
import moveit_commander
import tf
import rospy
import sys
import moveit_msgs.srv
import geometry_msgs.msg
import sensor_msgs.msg
import rospkg
import numpy
import actionlib_msgs.msg
import visualization_msgs.msg
import enum

class DensoInterface:
    def __init__(self, group_name, rate, debug=False):
        print('[SCRIPT] Init robot and load group {}'.format(group_name))
        self._init_config(rate, debug)
        self._init_robot(group_name)
        self._init_scene()
    
    def _init_config(self, rate, debug):
        """

        General configuration for ROS interfaces.

        @params
            rate: Integer
                Frequency in hertz for class update.
            debug: Boolean
                Configures whether debug messages should be printed.

        """
        self.debug = debug
        self.rate = rospy.Rate(rate)
        moveit_commander.roscpp_initialize(sys.argv)
        self.rospack = rospkg.RosPack()
        self.tf_listener = tf.TransformListener()
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.ik_service = rospy.ServiceProxy('compute_ik', moveit_msgs.srv.GetPositionIK, persistent=True)
        self.move_state_sub = rospy.Subscriber('/execute_trajectory/status', actionlib_msgs.msg.GoalStatusArray, self._state_callback)
        self.joint_state_pub = rospy.Publisher('/move_group/fake_controller_joint_states', sensor_msgs.msg.JointState, queue_size=10)
        self.marker_array_pub = rospy.Publisher("visualization_marker_array", visualization_msgs.msg.MarkerArray, queue_size=10)
        
    def _init_robot(self, group_name):
        """

        General robot configuration. Planning constraints and robot starting pose can be set here.
        
        @params
            group_name: String
                Name of the RViz move group to be considered. For now, there is only 'arm_group'.

        """
        self.robot = moveit_commander.RobotCommander()
        self.gripper_group = moveit_commander.MoveGroupCommander('gripper_group')
        self.gripper_group.set_planning_time(1)
        self.gripper_group.set_goal_position_tolerance(0.001)
        self.gripper_group.set_goal_joint_tolerance(0.005)
        self.gripper_group.set_max_acceleration_scaling_factor = 1.0
        self.gripper_group.set_max_velocity_scaling_factor = 1.0
        self.gripper_group.set_pose_reference_frame('simple_gripper_base')
        self.gripper_group.set_planner_id("BFMT")
        self.group = moveit_commander.MoveGroupCommander(group_name)
        self.group.set_planning_time(5)
        self.group.set_goal_position_tolerance(0.001)
        self.group.set_max_acceleration_scaling_factor = 1.0
        self.group.set_max_velocity_scaling_factor = 1.0
        self.group.set_pose_reference_frame('base_link')
        self.group.set_planner_id("BiTRRT")
        self.last_ik_position = [0, 0, 0]
        self.last_ik_orientation = [0, 0, 0, 0]

    def _init_scene(self):
        """

        General scene configuration. Default objects like boxes or meshes can be added here. For example:

        """
        self.scene = moveit_commander.PlanningSceneInterface()
        self.scene_objs = {}
        self.marker_array = visualization_msgs.msg.MarkerArray()
        self.update()
        rospy.sleep(1)

    def _state_callback(self, data):
        """

        Callback for trajectory action server messages, used in subscriber move_state_sub.
        
        @params
            data: actionlib_msgs.msg.GoalStatusArray
                Stores the statuses for goals that are currently being tracked by an action server.

        """
        if (len(data.status_list) > 0):
            self.move_state = data.status_list.pop().status

    def get_status(self):
        """
        Returns last status read from trajectory action server messages.

        @returns
            move_state: Integer
                Status based on actionlib_msgs/GoalStatus.msg enum. Possible states are: PENDING=0; 
                ACTIVE=1; PREEMPTED=2; SUCCEEDED=3; ABORTED=4; REJECTED=5; PREEMPTING=6; RECALLING=7;
                RECALLED=8; LOST=9.
                A detailed description can be found here:
                http://docs.ros.org/melodic/api/actionlib_msgs/html/msg/GoalStatus.html

        """
        return self.move_state

    def wait_trajectory_execution(self):
        """

        Wait until trajectory execution is not PENDING nor ACTIVE anymore.
        The 1 second sleep, at the end of the trajectory, prevent the robot from not finding next one.

        """
        while (not rospy.is_shutdown()):
            self.update()
            self.rate.sleep()
            if (self.get_status() >= actionlib_msgs.msg.GoalStatus.PREEMPTED):
                break
        rospy.sleep(1)

    def wait_for_state_update(self, object_name, object_is_attached=False, object_in_scene=False, timeout=4):
        """

        When adding a new object to the scene, it takes a little time to be avaliable for operations.
        Also, one object can be in none or only (and only) state from the following:
            - Attached to the robot, moving with it
            - Attached to a scene TF
        This function waits for recently added object to be updated before returning update success.
        More information can be found here:
        https://github.com/ros-planning/moveit_tutorials/blob/master/doc/move_group_python_interface/scripts/move_group_python_interface_tutorial.py

        @params
            object_name: String
                Name of the object that is being checked.
            object_is_attached: Boolean
                Indicates if desired state of the object is being attached to the robot.
            object_in_scene: Boolean
                Indicates if desired state of the object is attached to a scene TF.
            timeout: Integer
                Time in seconds of maximum waiting for desired state before returning False.

        @returns
            Boolean
                True if desired update of object was reached. False otherwise.

        """
        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
            # Check if object is attached
            attached_objects = self.scene.get_attached_objects().keys()
            is_attached = object_name in attached_objects

            # Check if object is in scene
            # Note that attaching the box will remove it from scene_objects
            scene_objects = self.scene.get_known_object_names()
            in_scene = object_name in scene_objects

            # Test if object is in the expected state
            if (object_is_attached == is_attached and object_in_scene == in_scene):
                print('[SCRIPT] {} state was updated!'.format(object_name))
                return True
            self.update()
            self.rate.sleep()
            seconds = rospy.get_time()

        # If timeout, return false
        print('[SCRIPT] Error! {} state was NOT updated!'.format(object_name))
        return False


    def add_box_to_scene(self, box_name, parent_link_name, box_position=[0,0,0], box_orientation=[0,0,0,1], box_size=1, box_color=[1, 1, 0, 0]):
        """

        Create a MoveIT collision box and a new TF assigned to it. TF information is stored in self.scene_objs dictionary.
        Also calls self.add_box_to_markers in order to create a visual representation.

        @params
            box_name: String
                Name that represents both the MoveIT collision box and the created TF.
            parent_link_name: String
                Name that represents parent TF for the new box.
            box_position: 3-element Float array
                XYZ position relative to the parent link.
            box_orientation: 4-element Float array
                Quaternion orientation relative to the parent link.
            box_size: Integer
                Size of the box in meters.
            box_color: 4-element Integer array
                Color of the box in the sequence [Alpha, Red, Green, Blue].

        """
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = parent_link_name
        box_pose.pose.position = geometry_msgs.msg.Point(box_position[0], box_position[1], box_position[2])
        box_pose.pose.orientation = geometry_msgs.msg.Quaternion(box_orientation[0], box_orientation[1], box_orientation[2], box_orientation[3])
        self.scene.add_box(box_name, box_pose, size=(box_size, box_size, box_size))
        self.scene_objs[box_name] = {
            'parent_link': parent_link_name,
            'position': box_pose.pose.position,
            'orientation': box_pose.pose.orientation
        }
        self.add_box_to_markers(box_name, parent_link_name, box_position, box_orientation, box_size, box_color)
        self.wait_for_state_update(box_name, object_in_scene=True, object_is_attached=False)
        print('[SCRIPT] Added obj {} to scene'.format(box_name))

    def add_box_to_markers(self, box_name, parent_link_name, box_position=[0,0,0], box_orientation=[0,0,0,1], box_size=1, box_color=[1, 1, 0, 0]):
        """

        Create a Marker box and attach it to self.marker_array.markers. This is the visual representation of a already existing MoveIT collision box.
        Assumes that a TF named box_name already exists.

        @params
            box_name: String
                Name for the Marker. Corresponds to a MoveIT object (and consequently to a TF from self.scene_objs).
            parent_link_name: String
                Name that represents parent TF for the new box.
            box_position: 3-element Float array
                XYZ position relative to the parent link.
            box_orientation: 4-element Float array
                Quaternion orientation relative to the parent link.
            box_size: Integer
                Size of the box in meters.
            box_color: 4-element Integer array
                Color of the box in the sequence [Alpha, Red, Green, Blue].

        """
        box = visualization_msgs.msg.Marker()
        box.id = len(self.marker_array.markers)
        box.header.stamp = self.now
        box.text = box_name
        box.type = visualization_msgs.msg.Marker.CUBE
        box.header.frame_id = parent_link_name
        box.action = visualization_msgs.msg.Marker.ADD
        box.pose.position.x = box_position[0]
        box.pose.position.y = box_position[1]
        box.pose.position.z = box_position[2]
        box.scale.x = box_size
        box.scale.y = box_size
        box.scale.z = box_size
        box.color.a = box_color[0]
        box.color.r = box_color[1]
        box.color.g = box_color[2]
        box.color.b = box_color[3]
        self.marker_array.markers.append(box)


    def add_mesh_to_scene(self, mesh_package_name, relative_path, mesh_name, parent_link_name, mesh_position=[0,0,0], mesh_orientation=[0,0,0,1], mesh_size=1, mesh_color=[1, 1, 1, 1]):
        """

        Create a MoveIT collision mesh and a new TF assigned to it. TF information is stored in self.scene_objs dictionary.
        Also calls self.add_mesh_to_markers in order to create a visual representation for it.

        @params
            mesh_package_name: String
                The package where the mesh object is stored. Example: vp6242b_description.
            relative_path: String
                The relative path from package root of the object. Example: /meshes/visual/table.dae.
            mesh_name: String
                Name of the object to be created. Example: table
            parent_link_name: String
                Name that represents parent TF for the new mesh.
            mesh_position: 3-element Float array
                XYZ position relative to the parent link.
            mesh_orientation: 4-element Float array
                Quaternion orientation relative to the parent link.
            mesh_size: Integer
                Scale of the mesh in meters.
            mesh_color: 4-element Integer array
                Color of the mesh in the sequence [Alpha, Red, Green, Blue].

        """    
        # Remove if already exists
        self.remove_object_from_scene(mesh_name)
        # First, get the full path from package
        full_mesh_path = self.rospack.get_path(mesh_package_name) + relative_path
        # MoveIT uses PoseStamped in order to publish 
        mesh_pose = geometry_msgs.msg.PoseStamped()
        mesh_pose.header.frame_id = parent_link_name
        mesh_pose.pose.position = geometry_msgs.msg.Point(mesh_position[0], mesh_position[1], mesh_position[2])
        mesh_pose.pose.orientation = geometry_msgs.msg.Quaternion(mesh_orientation[0], mesh_orientation[1], mesh_orientation[2], mesh_orientation[3])
        self.scene.add_mesh(mesh_name, mesh_pose, full_mesh_path, size=(mesh_size, mesh_size, mesh_size))
        self.scene_objs[mesh_name] = {
            'parent_link': parent_link_name,
            'position': mesh_pose.pose.position,
            'orientation': mesh_pose.pose.orientation
        }
        self.add_mesh_to_markers(
            mesh_package_name,
            relative_path,
            mesh_name,
            parent_link_name,
            mesh_color,
            mesh_position,
            mesh_orientation,
            mesh_size
        )
        print('[SCRIPT] Added obj {} to scene'.format(mesh_name))

    def add_mesh_to_markers(self, mesh_package_name, relative_path, mesh_name, parent_link_name, mesh_color=[1, 1, 1, 1], mesh_position=[0,0,0], mesh_orientation=[0,0,0,1], mesh_size=1):
        """

        Create a MoveIT collision mesh and a new TF assigned to it. TF information is stored in self.scene_objs dictionary.
        Also calls self.add_mesh_to_markers in order to create a visual representation for it.

        @params
            mesh_package_name: String
                The package where the mesh object is stored. Example: vp6242b_description
            relative_path: String
                The relative path from package root of the object. Example: /meshes/visual/table.dae
            mesh_name: String
                Name of the object to be created. Example: table
            parent_link_name: String
                Name that represents parent TF for the new mesh.
            mesh_position: 3-element Float array
                XYZ position relative to the parent link.
            mesh_orientation: 4-element Float array
                Quaternion orientation relative to the parent link.
            mesh_size: Integer
                Scale of the mesh in meters.
            mesh_color: 4-element Integer array
                Color of the mesh in the sequence [Alpha, Red, Green, Blue].

        """
        full_mesh_path = 'package://' + mesh_package_name + relative_path
        mesh = visualization_msgs.msg.Marker()
        mesh.text = mesh_name
        mesh.id = len(self.marker_array.markers)
        mesh.header.stamp = self.now
        mesh.header.frame_id = parent_link_name
        mesh.type = visualization_msgs.msg.Marker.MESH_RESOURCE
        mesh.mesh_resource = full_mesh_path
        mesh.action = visualization_msgs.msg.Marker.ADD
        mesh.pose.position.x = mesh_position[0]
        mesh.pose.position.y = mesh_position[1]
        mesh.pose.position.z = mesh_position[2]
        mesh.pose.orientation.x = mesh_orientation[0]
        mesh.pose.orientation.y = mesh_orientation[1]
        mesh.pose.orientation.z = mesh_orientation[2]
        mesh.pose.orientation.w = mesh_orientation[3]
        mesh.scale = geometry_msgs.msg.Vector3(mesh_size, mesh_size, mesh_size)
        mesh.color.a = mesh_color[0]
        mesh.color.r = mesh_color[1]
        mesh.color.g = mesh_color[2]
        mesh.color.b = mesh_color[3]
        mesh.lifetime = rospy.Duration(1)
        self.marker_array.markers.append(mesh)
    
    def add_tf_to_scene(self, tf_name, parent_link_name, tf_position, tf_orientation=[0,0,0,1]):
        """ 

        Create a new TF alone and store in self.scene_objs. Its parent either from the robot or scene.

        @params
            tf_name: String
                Name for the new TF.
            parent_link_name: String
                Name that represents parent TF.
            tf_position: 3-element Float array
                XYZ position relative to the parent link.
            tf_orientation: 4-element Float array
                Quaternion orientation relative to the parent link.

        """
        tf_pose = geometry_msgs.msg.PoseStamped()
        tf_pose.header.frame_id = parent_link_name
        tf_pose.pose.position = geometry_msgs.msg.Point(tf_position[0], tf_position[1], tf_position[2])
        tf_pose.pose.orientation = geometry_msgs.msg.Quaternion(tf_orientation[0], tf_orientation[1], tf_orientation[2], tf_orientation[3])
        self.scene_objs[tf_name] = {
            'parent_link': parent_link_name,
            'position': tf_pose.pose.position,
            'orientation': tf_pose.pose.orientation
        }
        print('[SCRIPT] Added tf {} to scene'.format(tf_name))

    def remove_object_from_scene(self, obj_name):
        """

        Remove a box or a mesh from scene.

        @params
            obj_name: String
                Name of object to be removed.

        """
        # Delete from TFs
        if (obj_name in self.scene_objs.keys()):
            del self.scene_objs[obj_name]
        # Delete from scene collision
        scene_objects = self.scene.get_known_object_names()
        if (obj_name in scene_objects):
            self.scene.remove_world_object(obj_name)
        # Search for names in marker_array
        for marker in filter(lambda x: (x.text == obj_name), self.marker_array.markers):
            self.marker_array.markers.remove(marker)
        

    def broadcast_tfs(self):
        """

        Publish all TF pose transformations stored in self.scene_objs.

        """
        for (key, obj) in zip(self.scene_objs.keys(), self.scene_objs.values()):
            self.tf_broadcaster.sendTransform(
                (obj['position'].x, obj['position'].y, obj['position'].z), 
                (obj['orientation'].x, obj['orientation'].y, obj['orientation'].z, obj['orientation'].w),
                self.now,
                key,
                obj['parent_link']
            )

    def update_markers(self):
        """

        Update each Marker element from self.marker_array.markers with TF poses from self.scene_objs dictionary.

        """
        for i in range(len(self.marker_array.markers)):
            marker_name = self.marker_array.markers[i].text
            if (marker_name in self.scene_objs.keys()):
                self.marker_array.markers[i].pose.position = self.scene_objs[marker_name]['position']
                self.marker_array.markers[i].pose.orientation = self.scene_objs[marker_name]['orientation']
                self.marker_array.markers[i].header.stamp = self.now
                self.marker_array.markers[i].header.frame_id = self.scene_objs[marker_name]['parent_link']
   
        self.marker_array_pub.publish( self.marker_array )

    def get_tf_transform(self, frame_name, frame_reference_name=None):
        """

        Get a TF from frame_reference_name to frame_name.

        @params
            frame_name: String
                Name of target frame.
            frame_reference_name: String
                Name of parent frame of reference.
        
        @returns
            2-element Tuple
                A tuple containing a 3-element Float position array and a 4-element Float orientation array.


        """
        if (not frame_reference_name):
            frame_reference_name = self.robot.get_planning_frame()
        try:
            last_time = self.tf_listener.getLatestCommonTime(frame_name, frame_reference_name)
            (trans, quat) = self.tf_listener.lookupTransform(frame_name, frame_reference_name, last_time)
            return (trans, quat)
        except Exception as e:
            print('[ERROR] Cannot find transform: {}'.format(e))
            return (None, None)


    def plan_trajectory(self, position=[0,0,0], orientation=[0, 0, 0, 1]):
        """

        Plan a trajectory from current wrist pose to desired wrist pose wrt base_link.
        

        @params
            position: 3-element Float array
                Desired position for the wirst (last link) of the robot.
            orientation: 4-element Float array
                Desired orientation for the wrist (last link) of the robot.

        """
        # Creating target_pose
        target_pose = geometry_msgs.msg.Pose()
        target_pose.position = geometry_msgs.msg.Point(position[0], position[1], position[2])
        target_pose.orientation = geometry_msgs.msg.Quaternion(orientation[0], orientation[1], orientation[2], orientation[3])
        self.group.set_pose_target(target_pose)
        # Profiling time
        print('[SCRIPT] Starting planning for new target pose: ')
        print(target_pose)
        now = rospy.Time.now()
        self.plan = self.group.plan()
        then = rospy.Time.now()
        print('[SCRIPT] Plan for took {} seconds'.format((then - now).to_sec()))
        # Checking if pose is reachable
        if (len(self.plan.joint_trajectory.points) == 0):
            print('[SCRIPT] Plan has failed. Pose is not reachable')
            self.group.clear_pose_targets()
        else:
            print('[SCRIPT] Plan was sucessful.')    
            self.target_pose = target_pose

    def plan_tool_trajectory(self, tool_frame, position=[0, 0, 0], orientation=[0, 0, 0, 1]):
        """
        Plan a trajectory from tool_frame pose to desired tool_frame pose wrt base_link.

        @params
            position: 3-element Float array
                Desired position for the tool_frame.
            orientation: 4-element Float array
                Desired orientation for the tool_frame.
        """                
        # Calculate translation and rotation from tool to wrist
        (tool_to_wrist_trans, tool_to_wrist_quat) = self.get_tf_transform(tool_frame, 'simple_gripper_base')
        if (tool_to_wrist_quat is None):
            return
        # Extract transformation matrix from tool to wrist 
        transf_mat = tf.transformations.compose_matrix(
            angles=tf.transformations.euler_from_quaternion(tool_to_wrist_quat),
            translate=tool_to_wrist_trans
        )
        # Calculate starting frame with passed parameters
        start_mat = tf.transformations.compose_matrix(
            angles=tf.transformations.euler_from_quaternion(orientation),
            translate=position,
        )
        # Calculate new goal frame
        goal_mat = numpy.dot(start_mat, transf_mat)
        goal_quat = tf.transformations.quaternion_from_matrix(goal_mat)
        goal_pos = tf.transformations.translation_from_matrix(goal_mat)
        # Call the planning for new goal
        self.plan_trajectory(goal_pos, goal_quat)

    def execute_trajectory(self, wait=False):
        """

        Try to execute previously planned trajectory. Not blocking by default.

        @params
            wait: Boolean
                If function will block other executions until finishes.

        """
        try:
            self.group.execute(self.plan, wait=False)
            if( wait ):
                self.wait_trajectory_execution()
        except Exception as e:
            print('[ERROR] Cannot execute trajectory: {}'.format(e))
            pass

    def attach_box(self, box_name):
        """

        Attach an already existing box to the end-effector frame.

        @params
            box_name: String
                Name of the box. Should exist as one of self.scene_objs key.

        """
        touch_links = self.robot.get_link_names(group='gripper_group')
        del self.scene_objs[box_name]
        eef_link = self.group.get_end_effector_link()
        eef_to_box_trans, eef_to_box_quat = self.get_tf_transform(eef_link, box_name)
        self.add_tf_to_scene(box_name, eef_link, eef_to_box_trans, eef_to_box_quat)
        self.scene.attach_box(eef_link, box_name, touch_links=touch_links)
        self.wait_for_state_update(box_name, object_is_attached=True, object_in_scene=False)

    def attach_mesh(self, mesh_name):
        """

        Attach an already existing mesh to the end-effector frame.

        @params
            obj_name: String
                Name of the object. Should exist as one of self.scene_objs key.

        """
        touch_links = self.robot.get_link_names(group='gripper_group')
        del self.scene_objs[mesh_name]
        eef_link = self.group.get_end_effector_link()
        eef_to_mesh_trans, eef_to_mesh_quat = self.get_tf_transform(eef_link, mesh_name)
        self.add_tf_to_scene(mesh_name, eef_link, eef_to_mesh_trans, eef_to_mesh_quat)
        self.scene.attach_mesh(eef_link, mesh_name, touch_links=touch_links)
        self.wait_for_state_update(mesh_name, object_is_attached=True, object_in_scene=False)

    def dettach_box(self, box_name):
        """

        Dettach an already existing box from the end-effector frame.

        @params
            box_name: String
                Name of the box. Should exist as one of self.scene_objs key.

        """
        del self.scene_objs[box_name]
        eef_link = self.group.get_end_effector_link()
        base_to_box_trans, base_to_box_quat = self.get_tf_transform('base_org', box_name)
        self.add_tf_to_scene(box_name, 'base_org', base_to_box_trans, base_to_box_quat)
        self.scene.remove_attached_object(eef_link, name=box_name)
        self.wait_for_state_update(box_name, object_is_attached=False, object_in_scene=True)

    def detach_all(self):
        """

        Dettach all objects from the end-effector frame.

        """
        attached_objects = self.scene.get_attached_objects().keys()
        for object_name in attached_objects:
            self.dettach_box(object_name)

    def update_gripper(self, joint_value, wait=False):
        """

        Updates gripper state.

        @params
            joint_value: Float
                How much should gripper close. From 0 (totally opened) to 0.83 (totally closed).

        """
        if (joint_value < 0):
            joint_value = 0
        elif (joint_value > 0.83):
            joint_value = 0.83

        joint_state_msg = sensor_msgs.msg.JointState()
        joint_state_msg.name.append('simple_gripper_right_driver_joint')
        joint_state_msg.position.append(joint_value)
        self.joint_state_pub.publish(joint_state_msg)
        if( wait ):
            rospy.sleep(1)

    def init_ik_mode(self):
        """

        Put robot in real_time inverse kinematics state. Does not use trajectory action server anymore.
        
        """
        rospy.wait_for_service('compute_ik')
        self.compute_ik = rospy.ServiceProxy('compute_ik', moveit_msgs.srv.GetPositionIK, persistent=True)
        self.service_request = moveit_msgs.msg.PositionIKRequest()
        self.service_request.group_name = 'arm_group'
        self.service_request.robot_state = self.robot.get_current_state()
        self.service_request.timeout.secs = 0.1
        self.service_request.avoid_collisions = True
    
    def move_to_pose(self, position, orientation, gripper_joint_state=0, base_frame='base_link'):
        """        

        Move robot to a new pose if different from the current one.

        @params
            position: 3-element Float array
                Desired position for the tool_frame.
            orientation: 4-element Float array
                Desired orientation for the tool_frame.
            gripper_joint_state: Float
                Value of gripper closure between 0 and 0.83.
            base_frame: String
                Name of reference frame for current pose.

        @returns
            Boolean
                Sucess of inverse kinematics motion
        
        """
        if ( numpy.array_equal(position, self.last_ik_position) and numpy.array_equal(orientation, self.last_ik_orientation) ):
            return True

        time_now = self.now
        pose_stamped = geometry_msgs.msg.PoseStamped()
        pose_stamped.header.frame_id = base_frame
        pose_stamped.header.stamp = time_now 
        pose_stamped.pose.position = geometry_msgs.msg.Point(position[0], position[1], position[2])
        pose_stamped.pose.orientation = geometry_msgs.msg.Quaternion(orientation[0], orientation[1], orientation[2], orientation[3])
        self.service_request.pose_stamped = pose_stamped
        response = self.ik_service(ik_request = self.service_request)
        # Check if a solution was found
        if ( response.error_code.val == 1 ):
            # Get joint values from result
            joint_goal = self.group.get_current_joint_values()
            joint_goal[0] = response.solution.joint_state.position[0]
            joint_goal[1] = response.solution.joint_state.position[1]
            joint_goal[2] = response.solution.joint_state.position[2]
            joint_goal[3] = response.solution.joint_state.position[3]
            joint_goal[4] = response.solution.joint_state.position[4]
            joint_goal[5] = response.solution.joint_state.position[5]
            # Define joint message
            joint_state_msg = sensor_msgs.msg.JointState()
            joint_state_msg.header.stamp = time_now
            joint_state_msg.name = response.solution.joint_state.name
            joint_state_msg.position = response.solution.joint_state.position
            joint_state_msg.velocity = []
            joint_state_msg.effort = []
            # Define gripper state
            gripper_joint_index = joint_state_msg.name.index('simple_gripper_right_driver_joint')
            temp_positions = list(joint_state_msg.position)
            temp_positions[gripper_joint_index] = gripper_joint_state
            joint_state_msg.position = tuple(temp_positions)
            # Publish message
            self.joint_state_pub.publish(joint_state_msg)
            return True
        else:
            return False

    def move_to_stored_pose(self, pose_name, wait=False):
        """
        Move to one of predefined poses.

        @params
            pose_name: String
                Name of the stored pose of robot
        """
        self.group.set_named_target(pose_name)
        self.plan = self.group.plan()
        self.execute_trajectory(wait=True)

    def get_close_transforms(self, tool_frame='tool_center', threshold=0.05):
        """

        Get all TFs that are close to tool_frame

        @params
            tool_frame: String
                Name of the tool_frame being analyzed.
            threshold: Float
                Radius in meters in order to a TF be considered close and be returned.

        """
        close_objects = []
        for object_name in self.scene_objs.keys():
            try:
                # Get last common time between transforms
                last_time = self.tf_listener.getLatestCommonTime(object_name, tool_frame)

                # Transformation to tool_center is discarded
                if( object_name == tool_frame ):
                    continue
                # Calculate transformation to object
                trans, _ = self.tf_listener.lookupTransform(object_name, tool_frame, last_time)
                trans_norm = tf.transformations.vector_norm(trans)
                # Compare norm of distance with threshold
                if (trans_norm < threshold):
                    close_objects.append((object_name, trans_norm))
            except Exception as e:
                print('[ERROR] Cannot find transform: {}'.format(e))
                pass
        # Sort objects by distance
        return sorted(close_objects, key=lambda x: x[1])
    
    
    def update(self):
        """

        Broadcast TFs and publish markers informations before get to the next time step.

        """
        self.update_markers()
        self.broadcast_tfs()
        self.now = rospy.Time.now()



    