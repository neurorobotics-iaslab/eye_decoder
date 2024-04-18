# header
std_msgs/Header header

# pupil positions
geometry_msgs/Point right_pupil
geometry_msgs/Point left_pupil

# radius of pupils
float32 right_radius
float32 left_radius

# nose axis
geometry_msgs/Point[] nose_points

# blinking
bool blink
int32 count_frame_blinking

# distance to nose axis
float32 distance_right
float32 distance_left

# two types of image
sensor_msgs/Image face_image