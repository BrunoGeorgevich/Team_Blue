#!/usr/bin/env python
import rospy
import tf
import numpy as np
from densointerface import DensoInterface
from geometry_msgs.msg import Quaternion

RATE = 30
NODE_NAME = 'pile_cubes'
DEFAULT_GROUP = 'arm_group'
BOX_SIZE = 0.04
OPEN_GRIPPER_JOINT = 0.0
CLOSE_GRIPPER_JOINT = 0.4



def run_node():
    rospy.init_node(NODE_NAME)
    rate = rospy.Rate(RATE)

    # Robot interface configs
    interface = DensoInterface(group_name='arm_group', rate=RATE)
    interface.add_box_to_scene('box_r', 'base_org', [0.25, -0.10, BOX_SIZE / 2], box_orientation=[0, 0, 0.6502878, -0.7596879], box_size=BOX_SIZE, box_color=[1, 1, 0, 0])
    interface.add_box_to_scene('box_g', 'base_org', [0.3,    0, BOX_SIZE / 2], box_size=BOX_SIZE, box_color=[1, 0, 1, 0])
    interface.add_box_to_scene('box_b', 'base_org', [0.35,  0.10, BOX_SIZE / 2], box_orientation=[0, 0, 0.4871745, -0.8733046], box_size=BOX_SIZE, box_color=[1, 0, 0, 1])
    interface.add_mesh_to_scene('vp6242b_description', '/meshes/visual/table.dae', 'table', 'world', mesh_orientation=[0, 0, 0.707, 0.707], mesh_color=[1, 0.984, 0.956, 0.796])
    interface.add_tf_to_scene('tool_center', 'simple_gripper_base', [0, 0, 0.14])
    interface.broadcast_tfs()
    rospy.sleep(1)

    # Testing going from cube to cube
    rot_cube_to_tool = tf.transformations.rotation_matrix(3.14, (0, 1, 0)) # R(Y, 180)
    quat_cube_to_tool = tf.transformations.quaternion_from_matrix(rot_cube_to_tool)

    def grab_cube(cube_name):
        goal_pos, goal_quat = interface.get_tf_transform('base_link', cube_name)
        goal_quat = np.array(goal_quat)
        goal_quat = tf.transformations.quaternion_multiply(goal_quat, quat_cube_to_tool)
        interface.plan_tool_trajectory('/tool_center', goal_pos , goal_quat)    
        interface.execute_trajectory(wait=True)
        interface.attach_box(cube_name)
        interface.update_gripper(CLOSE_GRIPPER_JOINT, wait=True)

    def pile_onto_cube(this_cube, onto_cube):
        goal_pos, goal_quat = interface.get_tf_transform('base_link', onto_cube)
        goal_quat = np.array(goal_quat)
        goal_quat = tf.transformations.quaternion_multiply(goal_quat, quat_cube_to_tool)
        interface.plan_tool_trajectory('/tool_center', [goal_pos[0], goal_pos[1], goal_pos[2] + BOX_SIZE] , goal_quat)    
        interface.execute_trajectory(wait=True)
        interface.update_gripper(OPEN_GRIPPER_JOINT, wait=True)
        interface.dettach_box(this_cube)

    def go_home():
        interface.group.set_named_target('home')
        interface.plan = interface.group.plan()
        interface.execute_trajectory(wait=True)    

    # Matin tasks
    grab_cube('box_r')
    pile_onto_cube(this_cube='box_r', onto_cube='box_g')
    grab_cube('box_b')
    pile_onto_cube(this_cube='box_b', onto_cube='box_r')
    go_home()

    # Main loop
    while(not rospy.is_shutdown()):
        try:
            interface.broadcast_tfs()
            rate.sleep()

        except rospy.ServiceException as e:
            print('[ERROR] Service call failed: {}'.format(e))
            
if __name__ == '__main__':
    try:
        run_node()
    except rospy.ROSInterruptException:
        rospy.signal_shutdown("KeyboardInterrupt")
