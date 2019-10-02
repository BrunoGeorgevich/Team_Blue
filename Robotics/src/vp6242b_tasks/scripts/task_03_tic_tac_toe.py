#!/usr/bin/env python
import rospy
import tf
import numpy as np
from densointerface import DensoInterface
from geometry_msgs.msg import Quaternion, Point
from std_msgs.msg import Int8MultiArray, MultiArrayDimension, Float64MultiArray, Float64
import math

#  Definitions
RATE = 60
INTERFACE = None
NODE_NAME = 'board_environment'
OPEN_GRIPPER_JOINT = 0.0
CLOSE_GRIPPER_JOINT = 0.65
GRIPPER_IS_CLOSED = False
COL_WIDTH = 0.06 + 0.005
ROW_HEIGHT = 0.06 + 0.005
LAST_BOARD_STATE =  np.empty((3, 3), dtype=object) # -1 is X, 1 is O, 0 is empty
NEW_BOARD_STATE =  np.empty((3, 3), dtype=object)  # -1 is X, 1 is O, 0 is empty
PROCESSED_NEW_BOARD_STATE = True                   # Only process new one if last was done processing
CELL_POSITIONS = np.empty((3, 3), dtype=object)    # Cell positions of form (X, Y) relative to board center
STARTED = False
MOVING = False
X_OFFSET = 0.12
Z_OFFSET = 0.12
SIDE_OFFSET = 0.12
SYMBOL_SIZE = 0.015

def size_cb(msg):
    global SYMBOL_SIZE
    SYMBOL_SIZE = msg.data

def offset_cb(msg):
    """
    Callback to the pen Z offset
    """
    global Z_OFFSET, X_OFFSET, CELL_POSITIONS, ROW_HEIGHT, COL_WIDTH, INTERFACE, SIDE_OFFSET
    X_OFFSET = msg.data[0]
    Z_OFFSET = msg.data[1]

    INTERFACE.add_mesh_to_scene('vp6242b_description', '/meshes/visual/white_board.dae', 'white_board', 'base_org', mesh_position=[0.35 + X_OFFSET, -SIDE_OFFSET, 0.02 + Z_OFFSET], mesh_color=[1, 1, 1, 1])
    INTERFACE.add_mesh_to_scene('vp6242b_description', '/meshes/visual/board_marks.dae', 'board_marks', 'base_org', mesh_position=[0.35 + X_OFFSET, -SIDE_OFFSET, 0.02 + Z_OFFSET], mesh_color=[1, 0, 0, 0])
    
    # Define cell positions
    for (i, X) in enumerate([-ROW_HEIGHT, 0, ROW_HEIGHT]):
        for (j, Y) in enumerate([-COL_WIDTH, 0, COL_WIDTH]):
            CELL_POSITIONS[i][j] = (X, Y + SIDE_OFFSET)
            cell_name = 'cell_{}_{}'.format(i, j)
            INTERFACE.add_tf_to_scene(cell_name, 'board_marks', tf_position=[X, Y+SIDE_OFFSET, 0])

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
            # Define mesh position
            x, y = CELL_POSITIONS[i][j]
            # Check if not empty
            if (cell_value == 1 or cell_value == -1):
                # Define new mesh model
                mesh_model = ('X' if cell_value == -1 else 'O')
                # Add new mesh
                INTERFACE.add_mesh_to_scene('vp6242b_description', '/meshes/visual/{}.dae'.format(mesh_model), cell_name, 'board_marks', mesh_position=[x, y, 0], mesh_color=[1, 0, 0, 0])
            else:
                INTERFACE.remove_object_from_scene(cell_name)
                INTERFACE.add_tf_to_scene(cell_name, 'board_marks', tf_position=[x, y, 0])
    # Allow new messages to be analyzed
    PROCESSED_NEW_BOARD_STATE = True

def draw_X(row, column):
    """
    Draw a X in given board position

    @params
        position : tuple
            Row and column where the X will be drawn.
    """
    global CELL_POSITIONS, INTERFACE, MOVING, SYMBOL_SIZE
    # Define name of cell TF
    cell_name = 'cell_{}_{}'.format(row, column)
    # Get transformation from base_link to cell_name object
    position, _ = INTERFACE.get_tf_transform('base_link', frame_reference_name=cell_name)
    # Create an array of points around cell position with fixed orientation
    orientation = [0.707, 0, 0, 0.707]
    poses = []
    offset_low = 0.08
    offset_high = 0.10
    # First line
    poses.append( ([position[0] - SYMBOL_SIZE , position[1] - SYMBOL_SIZE, position[2] + offset_high], orientation) )
    poses.append( ([position[0] - SYMBOL_SIZE , position[1] - SYMBOL_SIZE, position[2] + offset_low], orientation) )
    poses.append( ([position[0] + SYMBOL_SIZE , position[1] + SYMBOL_SIZE, position[2] + offset_low], orientation) )
    poses.append( ([position[0] + SYMBOL_SIZE , position[1] + SYMBOL_SIZE, position[2] + offset_high], orientation) )
    # Second line
    poses.append( ([position[0] - SYMBOL_SIZE , position[1] + SYMBOL_SIZE, position[2] + offset_high], orientation) )
    poses.append( ([position[0] - SYMBOL_SIZE , position[1] + SYMBOL_SIZE, position[2] + offset_low], orientation) )
    poses.append( ([position[0] + SYMBOL_SIZE , position[1] - SYMBOL_SIZE, position[2] + offset_low], orientation) )
    poses.append( ([position[0] + SYMBOL_SIZE , position[1] - SYMBOL_SIZE, position[2] + offset_high], orientation) )
    # Move to starting point
    starting_position, _ = poses[0]
    INTERFACE.plan_tool_trajectory(tool_frame='tool_center', position=starting_position, orientation=orientation)
    INTERFACE.execute_trajectory(wait=True)

    # Try to plan trajectory    
    fraction = INTERFACE.plan_multipoint_catesian_tool_trajectory('tool_center', poses)
    print('[SCRIPT] Completed {} \% of trajectory.'.format(fraction * 100))
    if (fraction == 1):
        INTERFACE.execute_trajectory(wait=True)
        INTERFACE.move_to_stored_pose('home')
    MOVING = False

def draw_O(row, column):
    """
    """
    global CELL_POSITIONS, INTERFACE, MOVING, SYMBOL_SIZE
    # Define name of cell TF
    cell_name = 'cell_{}_{}'.format(row, column)
    # Get transformation from base_link to cell_name object
    position, _ = INTERFACE.get_tf_transform('base_link', frame_reference_name=cell_name)
    # Create an array of points around cell position with fixed orientation
    orientation = [0.707, 0, 0, 0.707]
    poses = []
    offset_low = 0.08
    offset_high = 0.10
    radius = 0.015
    x = radius * math.cos(0)
    y = radius * math.sin(0)
    # First line
    poses.append( ([position[0] + x , position[1] + y, position[2] + offset_high], orientation) )
    for theta in range(0, 400, 10):
        x = radius * math.cos(theta * math.pi / 180)
        y = radius * math.sin(theta * math.pi / 180)
        poses.append( ([position[0] + x , position[1] + y, position[2] + offset_low], orientation) )

    # Move to starting point
    starting_position, _ = poses[0]
    INTERFACE.plan_tool_trajectory(tool_frame='tool_center', position=starting_position, orientation=orientation)
    INTERFACE.execute_trajectory(wait=True)

    # Try to plan trajectory    
    fraction = INTERFACE.plan_multipoint_catesian_tool_trajectory('tool_center', poses)
    print('[SCRIPT] Completed {} \% of trajectory.'.format(fraction * 100))
    if (fraction == 1):
        INTERFACE.execute_trajectory(wait=True)
        INTERFACE.move_to_stored_pose('home')
    MOVING = False

def command_cb(multiarray_data):
    """
    Callback for command messages.
    """
    global INTERFACE, MOVING
    if (MOVING):
        return
    MOVING = True
    row = multiarray_data.data[0]
    column = multiarray_data.data[1]
    if (row not in [0, 1, 2] or column not in [0, 1, 2]):
        print('[ERROR] Command outside allowed interval!')
        MOVING = False
        return
    # TODO: Create function draw_O
    # TODO: Check if symbol is X or O and call appropriate function
    # draw_X(row, column)
    draw_O(row, column)
    INTERFACE.move_to_stored_pose('home', wait=True)

def run_node():
    global RATE, NODE_NAME, OPEN_GRIPPER_JOINT, CLOSE_GRIPPER_JOINT, GRIPPER_IS_CLOSED, PROCESSED_NEW_BOARD_STATE, STARTED, INTERFACE, X_OFFSET, Z_OFFSET, SIDE_OFFSET
    # Init node
    rospy.init_node(NODE_NAME)
    rate = rospy.Rate(RATE)

    # Robot interface configs
    INTERFACE = DensoInterface(group_name='arm_group', rate=RATE)
    INTERFACE.add_mesh_to_scene('vp6242b_description', '/meshes/visual/table.dae', 'table', 'world', mesh_orientation=[0, 0, 0.7071, 0.7071], mesh_color=[1, 0.984, 0.956, 0.796])
    INTERFACE.add_mesh_to_scene('vp6242b_description', '/meshes/visual/white_board.dae', 'white_board', 'base_org', mesh_position=[0.35 + X_OFFSET, -SIDE_OFFSET, 0.02 + Z_OFFSET], mesh_color=[1, 1, 1, 1])
    INTERFACE.add_mesh_to_scene('vp6242b_description', '/meshes/visual/board_marks.dae', 'board_marks', 'base_org', mesh_position=[0.35 + X_OFFSET, -SIDE_OFFSET, 0.02 + Z_OFFSET], mesh_color=[1, 0, 0, 0])
    INTERFACE.add_tf_to_scene('tool_center', 'simple_gripper_base', [0, 0, 0.14], [ 0, -0.7068252, 0, 0.7073883 ])
    INTERFACE.update()

    # Move to home pose
    INTERFACE.move_to_stored_pose('home', wait=True)
    INTERFACE.wait_time(1)
    
    # Add pen to tool_frame and close the gripper
    INTERFACE.add_mesh_to_scene('vp6242b_description', '/meshes/visual/pen_only.dae', 'gripper_pen', 'tool_center', mesh_color=[1, 1, 1, 1])
    INTERFACE.update()

    # Attach pen to end effector and close the gripper
    INTERFACE.attach_mesh('gripper_pen')
    INTERFACE.update_gripper(CLOSE_GRIPPER_JOINT, wait=True)

    # Define cell positions
    for (i, X) in enumerate([-ROW_HEIGHT, 0, ROW_HEIGHT]):
        for (j, Y) in enumerate([-COL_WIDTH, 0, COL_WIDTH]):
            CELL_POSITIONS[i][j] = (X , Y+SIDE_OFFSET)
            cell_name = 'cell_{}_{}'.format(i, j)
            INTERFACE.add_tf_to_scene(cell_name, 'board_marks', tf_position=[X, Y+SIDE_OFFSET, 0])

    # Board state listener
    INTERFACE.update()
    STARTED = True
    rospy.Subscriber('board_state', Int8MultiArray, board_state_cb)
    rospy.Subscriber('arm_command', Int8MultiArray, command_cb)
    rospy.Subscriber('pen_offset', Float64MultiArray, offset_cb)
    rospy.Subscriber('symbol_size', Float64, size_cb)

    # TODO: Erase next lines. Only for tests
    # for i in range(3):
    #     for j in range(3):
    #         draw_X(i, j)
    # # Denso only reach first row... Need to create a smaller and closer table
    # INTERFACE.update()
    # print('[SCRIPT] Transform from base origin: {}'.format(INTERFACE.get_tf_transform('base_org', 'tool_center')))    
    # INTERFACE.move_to_stored_pose('home')

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
