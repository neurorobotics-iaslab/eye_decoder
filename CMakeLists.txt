cmake_minimum_required(VERSION 3.0.2)
project(eye_decoder)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  rosneuro_msgs
  sensor_msgs
  message_generation
)

add_message_files(
  FILES
    Eye.msg
)

generate_messages(
  DEPENDENCIES
    std_msgs
    sensor_msgs
  )

catkin_package(
  INCLUDE_DIRS 
  LIBRARIES 
  CATKIN_DEPENDS
  std_msgs
  message_runtime
  DEPENDS
)