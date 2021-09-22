import mediapipe as mp
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

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


def find_human_axis(keypoints):
    #axis of upper body. from hip to shoulders
    v1 = keypoints[12] - keypoints[24]
    v2 = keypoints[11] - keypoints[23]
    return (v1 + v2) / 2


def calibrate_human_axis():
    # have live webcam feed. stand upright and press q. the reference orientation(3d vector) of 'upright' is saved
    cap = cv.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if (not success):
            print('webcam not found')
            break
        
        results = inference(frame)
        if results.pose_landmarks is not None:
            plot_landmarks(frame, results)
            keypoints = results_to_keypoints(frame, results)
            reference_human_axis = find_human_axis(keypoints)
            color = (0, 255, 0)
            plot_line(frame, keypoints, reference_human_axis, color)
        
        cv.imshow('live', frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    cv.waitKey(1)
    return reference_human_axis


def fall_detection_human_axis(current_axis, reference_axis):
    #if current orientation is dissimiliar to calibrated reference oreintation -> fall detection
    if current_axis is None or reference_axis is None:
        return False
    similiarity = np.dot(current_axis, reference_axis) / (np.linalg.norm(current_axis) * np.linalg.norm(reference_axis))
    if similiarity < 0.5:
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


def plot_line(frame, keypoints, human_axis, color):
    startx = int(0.5 * (keypoints[24, 0] + keypoints[23, 0]))
    starty = int(0.5 * (keypoints[24, 1] + keypoints[23, 1]))
    start = (startx, starty)
    endx = int(startx + human_axis[0])
    endy = int(starty + human_axis[1])
    end = (endx, endy)
    thickness = 2
    cv.line(frame, start, end, color, thickness)


def plot_both_axes(current_human_axis, reference_human_axis):
    #3d matplotlib vector plot of reference axis and current axis
    cha = current_human_axis / np.linalg.norm(current_human_axis)
    rha = reference_human_axis / np.linalg.norm(reference_human_axis)
    ax.cla()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.view_init(elev=0, azim=270)
    ax.quiver(0, 0, 0, cha[0], cha[1], cha[2])
    ax.quiver(0, 0, 0, rha[0], rha[1], rha[2])
    plt.pause(0.0001)


##################################################################################################


###################################### main loop ##################################################
reference_human_axis = calibrate_human_axis()

cap = cv.VideoCapture(0)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
while True:
    success, frame = cap.read()
    if (not success):
        print('webcam not found')
        continue
    results = inference(frame)
    if results.pose_landmarks is not None:
        plot_landmarks(frame, results)
        keypoints = results_to_keypoints(frame, results)
        current_human_axis = find_human_axis(keypoints)
        
        detected = fall_detection_human_axis(current_human_axis, reference_human_axis)
        if detected:
            color = (0, 0, 255)
            text = 'fall'
        else:
            color = (0, 255, 0)
            text = 'not fall'
        plot_text(frame, text, color)
        plot_line(frame, keypoints, current_human_axis, color)

    cv.imshow('live', frame)

    if not (current_human_axis is None):
        plot_both_axes(current_human_axis, reference_human_axis)
        
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
cv.waitKey(1)
