import mediapipe as mp
import cv2 as cv
import numpy as np

######### init mediapipe solution ##################
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5)


####################################################

#################### fall detection functions ###########################################
def inference(frame):
    # run mediapipe net on frame and output results
    results = pose.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    return results


def results_to_keypoints(frame, results):
    # output: numpy array with shape=[33,3] 33 keypoints, x,y and z
    h, w, c = frame.shape
    keypoints = np.zeros((33, 3))
    landmark_list = results.pose_landmarks.landmark
    for i, landmark in enumerate(landmark_list):
        keypoints[i, 0] = landmark.x * w
        keypoints[i, 1] = landmark.y * h
        keypoints[i, 2] = landmark.z * w
    return keypoints


def find_upper_body(keypoints):
    v1 = keypoints[12] - keypoints[24]
    v2 = keypoints[11] - keypoints[23]
    return (v1+v2)/2

def find_left_leg(keypoints):
    return keypoints[23] - keypoints[27]

def find_right_leg(keypoints):
    return keypoints[24] - keypoints[28]

def find_orientation(line):
    orientation = -line[1] / np.abs(line[0])
    return orientation

def fall_detection_lines(upper_body_is_not_upright, left_leg_is_not_upright, right_leg_is_not_upright):
    fall_votes = 0
    if upper_body_is_not_upright:
        fall_votes += 1
    if left_leg_is_not_upright:
        fall_votes += 1
    if right_leg_is_not_upright:
        fall_votes += 1
    if fall_votes >= 2:
        return True
    else:
        return False


################################################################################################

############ helper functions plotting #########################################################################
def plot_landmarks(frame, results):
    mp_drawing.draw_landmarks(
                              frame,
                              results.pose_landmarks,
                              mp_pose.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                              )


def plot_text(frame, text, color):
    font = cv.FONT_HERSHEY_SIMPLEX
    position = (50, 50)
    fontScale = 1
    thickness = 2
    cv.putText(frame, text, position, font, fontScale, color, thickness, cv.LINE_AA)


def plot_upper_body(frame, keypoints, vector, color):
    startx = int(0.5 * (keypoints[24, 0] + keypoints[23, 0]))
    starty = int(0.5 * (keypoints[24, 1] + keypoints[23, 1]))
    start = (startx, starty)
    endx = int(startx + vector[0])
    endy = int(starty + vector[1])
    end = (endx, endy)
    thickness = 2
    cv.line(frame, start, end, color, thickness)


def plot_left_leg(frame, keypoints, vector, color):
    startx = int(keypoints[27, 0])
    starty = int(keypoints[27, 1])
    start = (startx, starty)
    endx = int(startx + vector[0])
    endy = int(starty + vector[1])
    end = (endx, endy)
    thickness = 2
    cv.line(frame, start, end, color, thickness)


def plot_right_leg(frame, keypoints, vector, color):
    startx = int(keypoints[28, 0])
    starty = int(keypoints[28, 1])
    start = (startx, starty)
    endx = int(startx + vector[0])
    endy = int(starty + vector[1])
    end = (endx, endy)
    thickness = 2
    cv.line(frame, start, end, color, thickness)


##################################################################################################


###################################### main loop ##################################################

cap = cv.VideoCapture(0)
while True:
    success, frame = cap.read()
    if (not success):
        print('webcam not found')
        continue
    results = inference(frame)
    if results.pose_landmarks is not None:
        plot_landmarks(frame, results)
        keypoints = results_to_keypoints(frame, results)
        upper_body = find_upper_body(keypoints)
        left_leg = find_left_leg(keypoints)
        right_leg = find_right_leg(keypoints)
        
        upper_body_orientation = find_orientation(upper_body)
        left_leg_orientation = find_orientation(left_leg)
        right_leg_orientation = find_orientation(right_leg)
        
        upper_body_is_not_upright = upper_body_orientation < 1
        if upper_body_is_not_upright:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        plot_upper_body(frame, keypoints, upper_body, color)

        left_leg_is_not_upright = left_leg_orientation < 1
        if left_leg_is_not_upright:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        plot_left_leg(frame, keypoints, left_leg, color)
        
        right_leg_is_not_upright = right_leg_orientation < 1
        if right_leg_is_not_upright:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        plot_right_leg(frame, keypoints, right_leg, color)
        
        
        fall_detected = fall_detection_lines(upper_body_is_not_upright, left_leg_is_not_upright, right_leg_is_not_upright)
        if fall_detected:
            color = (0, 0, 255)
            text = 'Fall'
        else:
            color = (0, 255, 0)
            text = 'Kein Fall'
        plot_text(frame, text, color)


    cv.imshow('live', frame)

    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
cv.waitKey(1)

