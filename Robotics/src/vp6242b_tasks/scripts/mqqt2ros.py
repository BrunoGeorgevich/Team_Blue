import paho.mqtt.client as mqtt_client
import time
import rospy
from std_msgs.msg import Int8MultiArray, MultiArrayDimension, Float64MultiArray, Float64
import numpy as np

BROKER_ADDRESS = '10.0.0.101'
ROS_NODE_NAME = 'mqqt_ros_bridge'

BOARD_STATE_TOPIC = 'board_state'
BOARD_STATE_ROS_TOPIC = 'board_state'
BOARD_STATE_ROS_PUBLISHER = None
BOARD_STATE_CLIENT_ID = 'board_state_server'

MOVE_TOPIC = 'move'
MOVE_ROS_TOPIC = 'arm_command'
MOVE_ROS_PUBLISHER = None
MOVE_CLIENT_ID = 'move_server'



def UNUSED(x):
    return x

def on_board_state_changed(client, userdata, message):
    UNUSED(client)
    UNUSED(userdata)
    msg = str(message.payload.decode("utf-8"))
    values = np.array(msg.split(','), dtype=np.int8)
    assert values.shape == (9,)
    state = Int8MultiArray()
    state.data = list(values)
    BOARD_STATE_ROS_PUBLISHER.publish(state)
    print('BOARD STATE with shape {} has values: {}'.format(values.shape, msg) )

def on_move_emitted(client, userdata, message):
    UNUSED(client)
    UNUSED(userdata)
    msg = str(message.payload.decode("utf-8"))
    values = np.array(msg.split(','), dtype=np.int8)
    assert values.shape == (2,)
    state = Int8MultiArray()
    state.data = list(values)
    MOVE_ROS_PUBLISHER.publish(state)
    print('MOVE with shape {} has values: {}'.format(values.shape, msg) )

def create_client(CLIENT_ID, BROKER_ADDRESS, MQTT_TOPIC, callback_fn):
    # is_connected = False
    client = mqtt_client.Client(CLIENT_ID)
    client.on_message = callback_fn
    client.connect(BROKER_ADDRESS)
    client.loop_start()
    client.subscribe(MQTT_TOPIC)
    return client

board_state_client = create_client(BOARD_STATE_CLIENT_ID, BROKER_ADDRESS, BOARD_STATE_TOPIC, on_board_state_changed)
move_client = create_client(MOVE_CLIENT_ID, BROKER_ADDRESS, MOVE_TOPIC, on_move_emitted)

try:
    rospy.init_node(ROS_NODE_NAME)
    rate = rospy.Rate(10)
    BOARD_STATE_ROS_PUBLISHER = rospy.Publisher(BOARD_STATE_ROS_TOPIC, Int8MultiArray, queue_size=1)
    MOVE_ROS_PUBLISHER = rospy.Publisher(MOVE_ROS_TOPIC, Int8MultiArray, queue_size=1)
    while True:
        time.sleep(0.1)
        rate.sleep()

except KeyboardInterrupt:
    print("exiting")
    client.disconnect()
    client.loop_stop()
# %%
