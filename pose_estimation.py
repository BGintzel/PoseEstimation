import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# global  ####################################################################################

boxes = deque(10 * [[0, 0, 0]], 10)


##############################################################################################

# helper functions plotting ##################################################################


def plot_text(frame, text, color, pos=(50, 50)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = pos
    fontScale = 1
    thickness = 2
    frame = cv2.putText(frame, text, position, font, fontScale, color, thickness, cv2.LINE_AA)
    return frame


def plot_upper_body(frame, keypoints, vector, color):
    startx = int(0.5 * (keypoints[24, 0] + keypoints[23, 0]))
    starty = int(0.5 * (keypoints[24, 1] + keypoints[23, 1]))
    start = (startx, starty)
    endx = int(startx + vector[0])
    endy = int(starty + vector[1])
    end = (endx, endy)
    thickness = 2
    cv2.line(frame, start, end, color, thickness)


def plot_left_leg(frame, keypoints, vector, color):
    startx = int(keypoints[27, 0])
    starty = int(keypoints[27, 1])
    start = (startx, starty)
    endx = int(startx + vector[0])
    endy = int(starty + vector[1])
    end = (endx, endy)
    thickness = 2
    cv2.line(frame, start, end, color, thickness)


def plot_right_leg(frame, keypoints, vector, color):
    startx = int(keypoints[28, 0])
    starty = int(keypoints[28, 1])
    start = (startx, starty)
    endx = int(startx + vector[0])
    endy = int(starty + vector[1])
    end = (endx, endy)
    thickness = 2
    cv2.line(frame, start, end, color, thickness)


def setup_img(img_loc, start, end, lm_list, box_fall, lines_fall, lm_fall, fall_value):
    if len(lm_list) > 1:
        if box_fall > 0.5:
            text = "Box: gefallen  " + str(box_fall)
            color = (0, 0, 255)
        else:
            text = "Box: Steht noch  " + str(box_fall)
            color = (0, 255, 0)
        img_loc = plot_text(img_loc, text, color, (20, 30))

        if lines_fall:
            text = "Lines: gefallen  " + str(lines_fall)
            color = (0, 0, 255)
        else:
            text = "Lines: Steht noch  " + str(lines_fall)
            color = (0, 255, 0)
        img_loc = plot_text(img_loc, text, color, (20, 60))

        if lm_fall > 0.5:
            text = "LM: gefallen  " + str(lm_fall)
            color = (0, 0, 255)
        else:
            text = "LM: Steht noch " + str(lm_fall)
            color = (0, 255, 0)
        img_loc = plot_text(img_loc, text, color, (20, 90))

        if fall_value > 0.5:
            text = "Overall: gefallen  " + str(fall_value)
            color = (0, 0, 255)
        else:
            text = "Overall: Steht noch " + str(fall_value)
            color = (0, 255, 0)
        img_loc = plot_text(img_loc, text, color, (20, 120))

        if boxes[-1][2]:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        img_loc = cv2.rectangle(img_loc, start, end, color, thickness=3)
        # img_loc = plot_text(img_loc, text, color, (20, 200))

    return img_loc


##############################################################################################

# fall detection functions ###################################################################

def calculate_fall_value(box_fall, lines_fall, lm_fall, box_weighting=1 / 3, lines_weighting=1 / 3, lm_weighting=1 / 3):
    lines = 0.0
    if lines_fall:
        lines = 1.0

    return box_fall * box_weighting + lines * lines_weighting + lm_fall * lm_weighting


def start_detection(results_loc, img_loc):
    global boxes
    fall_value = 0
    if results.pose_landmarks:
        lm_list = get_list(results_loc, img_loc)

        # bounding box: Value [0,1] ###
        start, end = get_bounding_box(lm_list)
        box_fall = detect_box_fall(start, end)

        # lines: Bool-Value  ###
        lines_fall = detect_lines_fall(results_loc, img_loc)

        # landmarks: Value [0,1]  ###
        lm_fall = detect_lm_fall(lm_list, img)

        fall_value = calculate_fall_value(box_fall, lines_fall, lm_fall)

        img_loc = setup_img(img_loc, start, end, lm_list, box_fall, lines_fall, lm_fall, fall_value)
        save_boxes(start, end)
    else:
        lm_list = []

    conf = calculate_confidence(img_loc, lm_list)

    return img_loc, fall_value, conf


def inference(frame):
    # run mediapipe net on frame and output results
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return results


def get_list(results_loc, img_loc):
    lm_list = []

    for id, lm in enumerate(results_loc.pose_landmarks.landmark):
        h, w, c = img_loc.shape
        x, y = int(lm.x * w), int(lm.y * h)
        if lm.visibility > 0.5 and 0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0:
            lm_list.append([id, x, y, lm.visibility])

    return lm_list


# fall detection bounding box ################################################################

def get_bounding_box(list_loc):
    x = []
    y = []
    start = (0, 0)
    end = (0, 0)
    for lm in list_loc:
        x.append(lm[1])
        y.append(lm[2])

    if len(y)>0 and len(x)>0:
        top = min(y)
        bot = max(y)
        left = min(x)
        right = max(x)
        start = (left, top)
        end = (right, bot)
    return start, end


def fallen(start, end):
    height = end[1] - start[1]
    width = end[0] - start[0]
    fallen = False
    if height/width < 1.3:
        fallen = True

    return fallen


def save_boxes(start, end):
    global boxes
    f = fallen(start, end)
    boxes.append([start, end, f])


def detect_box_fall(start, end):
    counter = 0

    for box in boxes:
        if box[2]:
            counter += 1

    fall_value = counter / len(boxes)

    return fall_value


# fall detection lines #######################################################################


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
    return (v1 + v2) / 2


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


def detect_lines_fall(results_loc, frame):
    keypoints = results_to_keypoints(frame, results_loc)
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

    return fall_detection_lines(upper_body_is_not_upright, left_leg_is_not_upright, right_leg_is_not_upright)


# fall detection landmarks ###################################################################

def detect_lm_fall(lm_list, img_loc):
    min = 10*(len(lm_list)/33)
    max_value = (len(lm_list)-min)/len(lm_list)
    h, w, c = img_loc.shape
    under_half = 0
    for lm in lm_list:
        if lm[2] > h * 0.5:
            under_half += 1

    fall_value = ((under_half - min) / len(lm_list))/max_value

    if fall_value < 0:
        fall_value = 0

    return fall_value


##################################################################################################


# confidence #####################################################################################


def check_lights(img_loc, threshold=20, minimum=0):
    img_gray = cv2.cvtColor(img_loc, cv2.COLOR_BGR2GRAY)
    average = img_gray.mean(axis=0).mean(axis=0)
    conf = (average - minimum) / (threshold-minimum)
    if conf > 1:
        conf = 1
    if conf < 0:
        conf = 0
    return conf


def calculate_confidence(img_loc, lm_list):

    conf = check_lights(img_loc)

    if len(lm_list) == 0:
        return conf
    else:
        return conf * len(lm_list) / 33


##################################################################################################


# main loop ######################################################################################


if __name__ == "__main__":

    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        if not success:
            print('webcam not found')
            continue

        results = inference(img)

        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        img, fall, confidence = start_detection(results, img)

        print("Gefallen: " + str(fall), ", Konfidenz: ", str(confidence))
        cv2.imshow("Image", img)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
