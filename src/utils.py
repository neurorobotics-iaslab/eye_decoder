import numpy as np

''' 
    calculate the distance given the 8 points for a eye
    input:
        points: the 8 points of the eye
    output:
        distance: the distance covered by the blinking
'''
def eye_ratio(points):

    P0, P3, P4, P5, P8, P11, P12, P13 = points

    numerator = (
        np.linalg.norm(P3 - P13) ** 3
        + np.linalg.norm(P4 - P12) ** 3
        + np.linalg.norm(P5 - P11) ** 3
    )
    denominator = 3 * np.linalg.norm(P0 - P8) ** 3
    
    distance = numerator / denominator

    return distance

'''
    calculate the blinking ratio given the landmarks and the indexes of the eyes
    input: 
        landmarks: the landmarks of the face
        left_idxs: the indexes of the left eye
        right_idxs: the indexes of the right eye
    output:
        ratio: the blinking ratio
'''
def blinking_ratio(landmarks, left_idxs, right_idxs):
    
    right_eye_ratio = eye_ratio(landmarks[right_idxs])
    left_eye_ratio = eye_ratio(landmarks[left_idxs])

    ratio = (right_eye_ratio + left_eye_ratio + 1) / 2

    return ratio

''' 
    calculate the distance between the eye and the nose axis defined in the nose_points
    input:
        center_eye: the center of the eye
        nose_points: the two points defining the nose axis
    output:
        distance: the distance between the eye and the nose axis
'''
def distance_nose(center_eyes, nose_points):
    
    distances = []
    for i in range(0, len(center_eyes)):
        center_eye = center_eyes[i]
        nose1 = nose_points[0]
        nose2 = nose_points[1]

        axis_direction = nose2 - nose1

        c_to_n = center_eye - nose1

        proj_length = np.dot(c_to_n, axis_direction) / np.linalg.norm(axis_direction)
        proj_vector = proj_length * axis_direction / np.linalg.norm(axis_direction)

        distance = np.linalg.norm(c_to_n - proj_vector)
        
        distances.append(distance)
    
    return distances