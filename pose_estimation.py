import mediapipe as mp
import cv2 as cv
import numpy as np
import time

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
    #scalar from [-1,1] -1 if upright, 0 if horizontal, 1 if upside down
    orientation = np.arctan2(line[1] , np.abs(line[0])) / (np.pi/2)
    return orientation

def fall_detection_lines(keypoints):
    #returns scalar from [0,1], 0 if upright, 1 if horizontal or upside down
    #based on orientation of both legs and upper body
    upper_body = find_upper_body(keypoints)
    left_leg = find_left_leg(keypoints)
    right_leg = find_right_leg(keypoints)
    
    upper_body_orientation = find_orientation(upper_body)
    left_leg_orientation = find_orientation(left_leg)
    right_leg_orientation = find_orientation(right_leg)
    
    upper_body_is_not_upright = upper_body_orientation > -0.5
    if upper_body_is_not_upright:
        color = (0, 0, 255)
    else:
        color = (0, 255, 0)
    plot_upper_body(frame, keypoints, upper_body, color)

    left_leg_is_not_upright = left_leg_orientation > -0.5
    if left_leg_is_not_upright:
        color = (0, 0, 255)
    else:
        color = (0, 255, 0)
    plot_left_leg(frame, keypoints, left_leg, color)
    
    right_leg_is_not_upright = right_leg_orientation > -0.5
    if right_leg_is_not_upright:
        color = (0, 0, 255)
    else:
        color = (0, 255, 0)
    plot_right_leg(frame, keypoints, right_leg, color)
    
    
    fall_value = (1/3)*(upper_body_orientation + left_leg_orientation + right_leg_orientation)
    fall_value = fall_value+1
    fall_value = np.clip(fall_value,0,1)
    
    return fall_value

def fall_detection_lower_half(keypoints, frame):
    #count landmarks in lower half of frame. head has higher weight than hands
    #feet should not be counted
    h,w,c = frame.shape
    fall_value = 0
    for i,kp in enumerate(keypoints[:25]):
        if kp[1] > h/2:
            if i in [15,16,17,18,19,20,21,22]:
                fall_value += 0.25/19
            else:
                fall_value += 1/19

    return fall_value

########### confidence functions ######################

def brightness(frame):
    intensities = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    mean_intensity = np.mean(intensities.flatten())
    return mean_intensity/255

def confidence_brightness(frame):
    #scalar from [0,1] 0 if dark 1 if bright
    #scaling determined through testing
    b = brightness(frame)
    confidence = (b-0.03)/ 0.15
    confidence = np.clip(confidence,0,1)
    return confidence

############################################################

###### Kalman Filter ########################################
#filters outliers, tracked value has inertia/delay, only when lots of frames vote for fall
class KalmanFilter:
    def __init__(self, fall_value, confidence):
        velocity = 0
        self.x = np.array([fall_value, velocity])
        self.I = np.array([
                           [confidence,0],
                           [0, 1/1000]
                           ])
        self.prev_time_stamp = time.time()
    
    def iterate(self, new_fall_value, new_confidence):
        #prediction
        next_time_stamp = time.time()
        delta_t = next_time_stamp - self.prev_time_stamp
        F = np.array([
                      [1, delta_t],
                      [0, 1]
                      ])
        #low sigma->high inertia
        Sigma = 0.9
        D = Sigma**2 * np.array([
                                   [1/4*delta_t**4, 1/2*delta_t**3],
                                   [1/2*delta_t**3, delta_t**2]
                                   ])
        self.prev_time_stamp = next_time_stamp
        self.x = F@self.x
        self.I = np.linalg.inv(F @ np.linalg.inv(self.I) @ F.T + D)
          
          
        #filtering
        new_x = np.array([new_fall_value, 0])
        new_I = np.array([
                            [new_confidence, 0],
                            [0, 1/1000]
                            ])
        filtered_I = self.I + new_I
        filtered_x = np.linalg.inv(filtered_I)@(self.I@self.x + new_I@new_x)
          
        self.x = filtered_x
        self.I = filtered_I


################################################################################################

############ helper functions plotting #########################################################################
def plot_landmarks(frame, results):
    mp_drawing.draw_landmarks(
                              frame,
                              results.pose_landmarks,
                              mp_pose.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                              )


def plot_text(frame, text, color=(0,255,0), position = (30, 50)):
    font = cv.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7
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

KF = KalmanFilter(0,1/1000)
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
        
        fall_value = fall_detection_lines(keypoints)
        if fall_value > 0.5:
            color = (0, 0, 255)
            text = 'Fall'
        else:
            color = (0, 255, 0)
            text = 'Kein Fall'
        plot_text(frame, str(round(fall_value,2)), color)
        
        KF.iterate(fall_value, 1)
        fall_state = KF.x[0]
        fall_state = np.clip(fall_state, 0, 1)
        if fall_state > 0.5:
            color = (0, 0, 255)
            text = 'Fall'
        else:
            color = (0, 255, 0)
            text = 'Kein Fall'
        plot_text(frame, str(round(fall_state,2)), color, (30,100))

    cv.imshow('live', frame)

    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
cv.waitKey(1)

