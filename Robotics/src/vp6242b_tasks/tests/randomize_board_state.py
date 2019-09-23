#!/usr/bin/env python
import rospy
import tf
import numpy as np
from std_msgs.msg import Int8MultiArray, MultiArrayDimension
from random import randint

def run_node():
    NODE_NAME = 'random_board_state_sender'
    TOPIC_NAME = 'board_state'
    RATE = 0.2

    # Init node
    rospy.init_node(NODE_NAME)
    rate = rospy.Rate(RATE)
    state_pub = rospy.Publisher(TOPIC_NAME, Int8MultiArray, queue_size=1)

    def publish_random_state():
        state = Int8MultiArray()
        state.layout.dim.append(MultiArrayDimension())
        state.layout.dim[0].label = 'rows'
        state.layout.dim[0].size = 3     # Number of rows
        state.layout.dim[0].stride = 9   # Size of whole data structure from here
        state.layout.dim.append(MultiArrayDimension())
        state.layout.dim[1].label = 'cols'
        state.layout.dim[1].size = 3   # Size of whole data structure from here
        state.layout.dim[1].stride = 3 # Number of cols
        state.data = [ randint(-1, 1) for i in range(9) ]
        print('[SCRIPT] Published {}'.format(state.data))
        state_pub.publish(state)

    # Main loop
    while(not rospy.is_shutdown()):
        try:
            publish_random_state()
            rate.sleep()
        except rospy.ServiceException as e:
            print('[ERROR] Service call failed: {}'.format(e))

if __name__ == '__main__':
    try:
        run_node()
    except rospy.ROSInterruptException:
        rospy.signal_shutdown("KeyboardInterrupt")