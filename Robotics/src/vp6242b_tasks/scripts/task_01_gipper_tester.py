#!/usr/bin/env python
import rospy
from densointerface import DensoInterface

RATE = 30
NODE_NAME = 'gripper_tester'
DEFAULT_GROUP = 'arm_group'

def run_node():
    rospy.init_node(NODE_NAME)
    rate = rospy.Rate(RATE)
    
    interface = DensoInterface(group_name='arm_group', rate=RATE)
    interface.add_tf_to_scene('tool_center', 'simple_gripper_base', [0, 0, 0.14])
    
    # Set starting gripper state to 0
    gripper_state = 0
    max_gripper_state = 0.83
    ds = 0.01

    #
    while(not rospy.is_shutdown()):
        try:
            # Broadcast transformations
            interface.broadcast_tfs()
            rate.sleep()
            
            # Update gripper state
            interface.update_gripper(gripper_state)
            gripper_state = gripper_state + ds

            # Change direction if extremes
            if (gripper_state >= max_gripper_state):
                gripper_state = max_gripper_state
                ds = -0.01
            elif (gripper_state <= 0):
                gripper_state = 0
                ds = 0.01


        except rospy.ServiceException as e:
            print('[ERROR] Service call failed: {}'.format(e))
            
if __name__ == '__main__':
    try:
        run_node()
    except rospy.ROSInterruptException:
        rospy.signal_shutdown("KeyboardInterrupt")
