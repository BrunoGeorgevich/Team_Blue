#!/usr/bin/env python
import rospy
import tf
import numpy as np
from densointerface import DensoInterface
from geometry_msgs.msg import Quaternion, Point
from std_msgs.msg import Int8MultiArray, MultiArrayDimension

#  Definitions
RATE = 60
INTERFACE = None
NODE_NAME = 'board_environment'
OPEN_GRIPPER_JOINT = 0.0
CLOSE_GRIPPER_JOINT = 0.6
GRIPPER_IS_CLOSED = False
COL_WIDTH = 0.137
ROW_HEIGHT = 0.096
LAST_BOARD_STATE =  np.empty((3, 3), dtype=object) # -1 is X, 1 is O, 0 is empty
NEW_BOARD_STATE =  np.empty((3, 3), dtype=object)  # -1 is X, 1 is O, 0 is empty
PROCESSED_NEW_BOARD_STATE = True                   # Only process new one if last was done processing
CELL_POSITIONS = np.empty((3, 3), dtype=object)     # Cell positions of form (X, Y)
STARTED = False

def board_state_cb(multiarray_data):
    """
    Callback for board state messages.
    """
    global NEW_BOARD_STATE, PROCESSED_NEW_BOARD_STATE
    if ( (PROCESSED_NEW_BOARD_STATE == False) or (STARTED == False) ):
        return
    PROCESSED_NEW_BOARD_STATE = False
    NEW_BOARD_STATE = np.array(multiarray_data.data).reshape((3, 3))

def process_board_state():
    """
    Function to update table in RViz
    """
    global LAST_BOARD_STATE, NEW_BOARD_STATE, PROCESSED_NEW_BOARD_STATE, INTERFACE
    # INTERFACE.update()
    for i in range(3):
        for j in range(3):
            # Define cell name
            cell_name = 'cell_{}_{}'.format(i, j)
            # Define new value of cell
            cell_value = NEW_BOARD_STATE[i][j]
            # Check if not empty
            if (cell_value == 1 or cell_value == -1):
                # Define new mesh model
                mesh_model = ('X' if cell_value == -1 else 'O')
                # Define mesh position
                x, y = CELL_POSITIONS[i][j]
                # Add new mesh
                INTERFACE.add_mesh_to_scene('vp6242b_description', '/meshes/visual/{}.dae'.format(mesh_model), cell_name, 'board_marks', mesh_position=[x, y, 0], mesh_color=[1, 0, 0, 0])
            else:
                INTERFACE.remove_object_from_scene(cell_name)
    # Allow new messages to be analyzed
    PROCESSED_NEW_BOARD_STATE = True

def run_node():
    global RATE, NODE_NAME, OPEN_GRIPPER_JOINT, CLOSE_GRIPPER_JOINT, GRIPPER_IS_CLOSED, PROCESSED_NEW_BOARD_STATE, STARTED, INTERFACE
    # Init node
    rospy.init_node(NODE_NAME)
    rate = rospy.Rate(RATE)

    # Robot interface configs
    INTERFACE = DensoInterface(group_name='arm_group', rate=RATE)
    INTERFACE.add_mesh_to_scene('vp6242b_description', '/meshes/visual/table.dae', 'table', 'world', mesh_orientation=[0, 0, 0.7071, 0.7071], mesh_color=[1, 0.984, 0.956, 0.796])
    INTERFACE.add_mesh_to_scene('vp6242b_description', '/meshes/visual/white_board.dae', 'white_board', 'base_org', mesh_position=[0.4, 0, 0.02], mesh_color=[1, 1, 1, 1])
    INTERFACE.add_mesh_to_scene('vp6242b_description', '/meshes/visual/board_marks.dae', 'board_marks', 'base_org', mesh_position=[0.4, 0, 0.02], mesh_color=[1, 0, 0, 0])
    INTERFACE.add_tf_to_scene('tool_center', 'simple_gripper_base', [0, 0, 0.14])
    INTERFACE.update()

    # Move to home pose
    INTERFACE.move_to_stored_pose('home', wait=True)

    # Add pen to tool_frame and close the gripper
    INTERFACE.add_mesh_to_scene('vp6242b_description', '/meshes/visual/pen_only.dae', 'gripper_pen', 'tool_center', mesh_orientation=[0, 0.7071, 0, -0.7071], mesh_color=[1, 1, 1, 1])
    INTERFACE.update()

    # Attach pen to end effector and close the gripper
    INTERFACE.attach_mesh('gripper_pen')
    INTERFACE.update_gripper(CLOSE_GRIPPER_JOINT, wait=True)

    # Define cell positions
    for (i, X) in enumerate([-ROW_HEIGHT, 0, ROW_HEIGHT]):
        for (j, Y) in enumerate([-COL_WIDTH, 0, COL_WIDTH]):
            CELL_POSITIONS[i][j] = (X, Y)

    # Board state listener
    INTERFACE.update()
    STARTED = True
    rospy.Subscriber('board_state', Int8MultiArray, board_state_cb)
    # Main loop
    while(not rospy.is_shutdown()):
        try:
            rate.sleep()
            INTERFACE.update()
            
            if (not PROCESSED_NEW_BOARD_STATE):
                process_board_state()

        except rospy.ServiceException as e:
            print('[ERROR] Service call failed: {}'.format(e))
            
if __name__ == '__main__':
    try:
        run_node()
    except rospy.ROSInterruptException:
        rospy.signal_shutdown("KeyboardInterrupt")
