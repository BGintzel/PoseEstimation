import socket
import json
import time
import multiprocessing
import select

from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from numpy.linalg import inv


##### Kalman #################################################################################

class KalmanFilter:
    def __init__(self, fall_value, confidence):
        velocity = 0
        self.x = np.array([fall_value, velocity])
        self.P = np.array([
            [1 / (confidence + 1e-6), 0],
            [0, 1000]
        ])
        self.prev_time_stamp = time.time()

    def iterate(self, new_fall_value, new_confidence):
        # prediction
        next_time_stamp = time.time()
        delta_t = next_time_stamp - self.prev_time_stamp
        F = np.array([
            [1, delta_t],
            [0, 1]
        ])
        # low sigma->high inertia
        Sigma = 0.9
        D = Sigma ** 2 * np.array([
            [1 / 4 * delta_t ** 4, 1 / 2 * delta_t ** 3],
            [1 / 2 * delta_t ** 3, delta_t ** 2]
        ])
        self.prev_time_stamp = next_time_stamp
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + D

        # filtering
        new_x = np.array([new_fall_value, 0])
        new_P = np.array([
            [1 / (new_confidence + 1e-6), 0],
            [0, 1000]
        ])
        filtered_P = inv(inv(self.P) + inv(new_P))
        filtered_x = filtered_P @ (inv(self.P) @ self.x + inv(new_P) @ new_x)

        self.x = np.clip(filtered_x, 0, 1)
        self.P = filtered_P


##############################################################################################

# global  ####################################################################################

boxes = deque(10 * [[0, 0, 0]], 10)
vid_detection_for_fusion = deque(10 * [[0.0, 0.0, 0.0]], 10)
vid_detection = deque(10 * [[0.0, 0.0, 0.0]], 10)
KF_lines = KalmanFilter(0, 0)
KF_box = KalmanFilter(0, 0)
KF_lm = KalmanFilter(0, 0)
fall_counter = 0
counter_locked = False


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


def setup_img(img_loc, start, end, lm_list, box_fall, lines_fall, lm_fall, fall_value, latest_fusion_values_loc):
    global fall_counter
    if len(lm_list) > 1:
        """
        if box_fall > 0.5:
            text = "Box: gefallen  " + str(round(box_fall,3))
            color = (0, 0, 255)
        else:
            text = "Box: Steht noch  " + str(round(box_fall,3))
            color = (0, 255, 0)
        img_loc = plot_text(img_loc, text, color, (20, 150))

        if lines_fall > 0.5:
            text = "Lines: gefallen  " + str(round(lines_fall,3))
            color = (0, 0, 255)
        else:
            text = "Lines: Steht noch  " + str(round(lines_fall,3))
            color = (0, 255, 0)
        img_loc = plot_text(img_loc, text, color, (20, 60))

        if lm_fall > 0.5:
            text = "LM: gefallen  " + str(round(lm_fall,3))
            color = (0, 0, 255)
        else:
            text = "LM: Steht noch " + str(round(lm_fall,3))
            color = (0, 255, 0)
        img_loc = plot_text(img_loc, text, color, (20, 150))


        if fall_value > 0.5:
            text = "Overall: gefallen  " + str(round(fall_value,3))
            color = (0, 0, 255)
        else:
            text = "Overall: Steht noch " + str(round(fall_value,3))
            color = (0, 255, 0)
        img_loc = plot_text(img_loc, text, color, (20, 120))"""

        if boxes[-1][2] > 0.5:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        img_loc = cv2.rectangle(img_loc, start, end, color, thickness=3)
        # img_loc = plot_text(img_loc, text, color, (20, 200))

    video_fall, video_confidence, radar_fall, radar_confidence, fusion_fall, fusion_confidence = latest_fusion_values_loc

    video_text = "Video   Fall: " + str(round(video_fall, 2)) + "  Conf: " + str(round(video_confidence, 2))
    radar_text = "Radar   Fall: " + str(round(radar_fall, 2)) + "    Conf: " + str(round(radar_confidence, 2))
    fusion_text = "Fusion   Fall: " + str(round(fusion_fall, 2)) + "    Conf: " + str(round(fusion_confidence, 2))

    if video_fall > 0.5:
        color = (0, 0, 255)
    else:
        color = (0, 255, 0)
    plot_text(img_loc, video_text, color, (20, 30))
    if radar_fall < 0:
        color = (0, 255, 255)
    elif radar_fall > 0.5:
        color = (0, 0, 255)
    else:
        color = (0, 255, 0)
    plot_text(img_loc, radar_text, color, (20, 60))
    if fusion_fall > 0.5:
        color = (0, 0, 255)
    else:
        color = (0, 255, 0)
    plot_text(img_loc, fusion_text, color, (20, 90))

    plot_text(img_loc, str(fall_counter), (255, 255, 255), (img_loc.shape[1] - 40, 30))

    return img_loc


##############################################################################################

# fall detection functions ###################################################################

def calculate_fall_value(box_fall, lines_fall, lm_fall, box_weighting=1 / 3, lines_weighting=1 / 3, lm_weighting=1 / 3):
    return box_fall * box_weighting + lines_fall * lines_weighting + lm_fall * lm_weighting


def start_detection(results_loc, img_loc, latest_fusion_values_loc):
    global boxes
    fall_value = 0
    start, end, box_fall, lines_fall, lm_fall = 0, 0, 0, 0, 0
    if results_loc.pose_landmarks:
        lm_list = get_list(results_loc, img_loc)

        # bounding box: Value [0,1] ###
        start, end = get_bounding_box(lm_list)
        box_fall = detect_box_fall(start, end)
        KF_box.iterate(box_fall, 1)
        box_fall = KF_box.x[0]

        # lines: Value [0,1]  ###
        lines_fall = fall_detection_lines(results_loc, img_loc)
        KF_lines.iterate(lines_fall, 1)
        lines_fall = KF_lines.x[0]

        # landmarks: Value [0,1]  ###
        lm_fall = detect_lm_fall(lm_list, img_loc)
        KF_lm.iterate(lm_fall, 1)
        lm_fall = KF_lm.x[0]

        fall_value = calculate_fall_value(box_fall, lines_fall, lm_fall)

        save_boxes(start, end)
    else:
        lm_list = []

    img_loc = setup_img(img_loc, start, end, lm_list, box_fall, lines_fall, lm_fall, fall_value,
                        latest_fusion_values_loc)

    conf = calculate_confidence(img_loc, lm_list)

    return img_loc, fall_value, conf


def inference(frame, pose):
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

    if len(y) > 0 and len(x) > 0:
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
    fallen = 0
    if width == 0:
        width = 1
    if height == 0:
        height = 1

    fallen = np.clip((width / height - 1 / 4) / (1.05), 0, 1)

    return fallen


def save_boxes(start, end):
    global boxes
    f = fallen(start, end)
    boxes.append([start, end, f])


def detect_box_fall(start, end):
    counter = 0

    # for box in boxes:
    # if box[2]:
    #    counter += 1

    # fall_value = counter / len(boxes)

    return boxes[-1][2]


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
    # scalar from [-1,1] -1 if upright, 0 if horizontal, 1 if upside down
    orientation = np.arctan2(line[1], np.abs(line[0])) / (np.pi / 2)
    return orientation


def fall_detection_lines(results, frame):
    keypoints = results_to_keypoints(frame, results)
    upper_body = find_upper_body(keypoints)
    left_leg = find_left_leg(keypoints)
    right_leg = find_right_leg(keypoints)

    upper_body_orientation = find_orientation(upper_body)
    left_leg_orientation = np.clip(find_orientation(left_leg), -1, 0.2)
    right_leg_orientation = np.clip(find_orientation(right_leg), -1, 0.2)

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

    fall_value = (1 / 3) * (upper_body_orientation + left_leg_orientation + right_leg_orientation)
    fall_value = fall_value + 1
    fall_value = np.clip(fall_value, 0, 1)

    return fall_value


# fall detection landmarks ###################################################################

def detect_lm_fall(lm_list, img_loc):
    fall_value = 0
    if len(lm_list) > 0:
        min = 10 * (len(lm_list) / 33)
        max_value = (len(lm_list) - min) / len(lm_list)
        h, w, c = img_loc.shape
        under_half = 0
        for lm in lm_list:
            if lm[2] > h * 0.5:
                under_half += 1

        fall_value = ((under_half - min) / len(lm_list)) / max_value

        if fall_value < 0:
            fall_value = 0

    return fall_value


##################################################################################################


# confidence #####################################################################################


def check_lights(img_loc, threshold=50, minimum=30):
    img_gray = cv2.cvtColor(img_loc, cv2.COLOR_BGR2GRAY)
    average = img_gray.mean(axis=0).mean(axis=0)
    # print(average)
    conf = (average - minimum) / (threshold - minimum)
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


def receive_per_udp(sock):
    timeout = 0.01
    inputs = [sock]
    outputs = []

    readable, writable, exceptional = select.select(inputs, outputs, inputs, timeout)
    if not readable:
        return None
    for readable_socket in readable:
        data, addr = readable_socket.recvfrom(1024)  # buffer size is 1024 bytes
        message_json = data.decode()
        message_dict = json.loads(message_json)
        radar_detection = message_dict
        radar_detection['timestamp'] = time.time()
    return list(radar_detection.values())


def append_new_fall_value():
    fall = 0
    confidence_loc = 0
    length_of_detection = 0
    for detection in vid_detection:
        if detection[0] != 0.0:
            fall += detection[1]
            confidence_loc += detection[2]
            length_of_detection += 1
    now = time.time()
    vid_detection_for_fusion.append([now, fall / length_of_detection, confidence_loc / length_of_detection])


def loop():
    UDP_IP = 'TODO'
    # UDP_IP = '10.249.54.144'
    UDP_PORT = 6789
    sock = socket.socket(socket.AF_INET,  # Internet
                         socket.SOCK_DGRAM)  # UDP
    sock.bind((UDP_IP, UDP_PORT))
    newest_radar_detection = [0, 0, 0]  # initialize with 0 until acutal data arrives

    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    cap = cv2.VideoCapture(0)
    current = time.time()
    latest_fusion_values = 0, 0, 0, 0, 0, 0
    # out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (1280, 720))

    while True:

        success, img = cap.read()
        if not success:
            print('webcam not found')
            continue

        results = inference(img, pose)

        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        img, fall, confidence = start_detection(results, img, latest_fusion_values)
        now = time.time()
        vid_detection.append([now, fall, confidence])

        radar_detection = receive_per_udp(sock)
        new_radar_is_valid = radar_detection is not None
        if radar_detection is not None:
            newest_radar_detection = radar_detection
            current = time.time()
        append_new_fall_value()
        latest_fusion_values = fusion(newest_radar_detection, img, new_radar_is_valid)

        update_counter(latest_fusion_values)
        # out.write(img)
        img = cv2.resize(img, (int(1000.0 * 1.333333), 1000))
        #print(img.shape)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    # out.release()


##################################################################################################


# fusion ######################################################################################

def fusion(radar_detection, img_loc, new_radar_is_valid):
    #print(time.time()-radar_detection[0])
    if not new_radar_is_valid and time.time()-radar_detection[0]>3:
        radar_detection[2] = 0
        radar_detection[1] = -1
    vid = vid_detection_for_fusion[-1]
    sum_of_confidences = 0
    confidence_fusion = 0
    fall_fusion = 0
    # print("vid: ", vid[0])
    # print("rad: ", radar_detection[0])
    # print(abs(vid[0] - radar_detection[0]))
    # wenn die letzen einträge weniger als eine Sekunde auseinander liegen
    sum_of_confidences = vid[2] + radar_detection[2]
    if sum_of_confidences != 0:
        g_vid = vid[2] / sum_of_confidences
        g_radar = radar_detection[2] / sum_of_confidences
        fall_fusion = g_vid * vid[1] + g_radar * radar_detection[1]
        confidence_fusion = sum_of_confidences / 2

        # print("Video   Fall: ", round(vid[1], 2), "  Conf: ", round(vid[2], 2))
        # print("Radar   Fall: ", round(radar_detection[1], 2), "    Conf: ", round(radar_detection[2], 2))
        # print("Fusion   Fall: ", round(fall_fusion, 2), "    Conf: ", round(confidence_fusion, 2))
        # print()
    return vid[1], vid[2], radar_detection[1], radar_detection[2], fall_fusion, confidence_fusion


def update_counter(latest_fusion_values_loc):
    global counter_locked, fall_counter
    video_fall, video_confidence, radar_fall, radar_confidence, fusion_fall, fusion_confidence = latest_fusion_values_loc
    if not counter_locked:
        if fusion_fall > 0.5 and fusion_confidence > 0:
            fall_counter += 1
            counter_locked = True
    else:
        if fusion_fall < 0.2:
            counter_locked = False


if __name__ == "__main__":
    loop()
