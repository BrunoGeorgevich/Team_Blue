#!/usr/bin/env python
import rospy
import tf
import numpy as np
from densointerface import DensoInterface
from geometry_msgs.msg import Quaternion, Point

def run_node():
    #  Definitions
    RATE = 60
    NODE_NAME = 'board_environment'
    OPEN_GRIPPER_JOINT = 0.0
    CLOSE_GRIPPER_JOINT = 0.6
    GRIPPER_IS_CLOSED = False

    # Init node
    rospy.init_node(NODE_NAME)
    rate = rospy.Rate(RATE)

    # Robot interface configs
    interface = DensoInterface(group_name='arm_group', rate=RATE)
    interface.add_mesh_to_scene('vp6242b_description', '/meshes/visual/table.dae', 'table', 'world', mesh_orientation=[0, 0, 0.7071, 0.7071], mesh_color=[1, 0.984, 0.956, 0.796])
    interface.add_mesh_to_scene('vp6242b_description', '/meshes/visual/white_board.dae', 'white_board', 'base_org', mesh_position=[0.4, 0, 0.02], mesh_color=[1, 1, 1, 1])
    interface.add_mesh_to_scene('vp6242b_description', '/meshes/visual/board_marks.dae', 'board_marks', 'base_org', mesh_position=[0.4, 0, 0.02], mesh_color=[1, 0, 0, 0])
    interface.add_tf_to_scene('tool_center', 'simple_gripper_base', [0, 0, 0.14])
    interface.update()

    # Move to home pose
    interface.move_to_stored_pose('home', wait=True)

    # Add pen to tool_frame and close the gripper
    interface.add_mesh_to_scene('vp6242b_description', '/meshes/visual/pen_only.dae', 'gripper_pen', 'tool_center', mesh_orientation=[0, 0.7071, 0, -0.7071], mesh_color=[1, 1, 1, 1])
    interface.update()

    # Attach pen to end effector and close the gripper
    interface.attach_mesh('gripper_pen')
    interface.update_gripper(CLOSE_GRIPPER_JOINT, wait=True)

    # Main loop
    while(not rospy.is_shutdown()):
        try:
            rate.sleep()
            interface.update()
            
        except rospy.ServiceException as e:
            print('[ERROR] Service call failed: {}'.format(e))
            
if __name__ == '__main__':
    try:
        run_node()
    except rospy.ROSInterruptException:
        rospy.signal_shutdown("KeyboardInterrupt")
