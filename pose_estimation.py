import cv2
import mediapipe as mp
from collections import deque

boxes = deque(10 * [[0, 0, 0]], 10)


def get_bounding_box(list_loc):
    x = []
    y = []
    for lm in list_loc:
        x.append(lm[1])
        y.append(lm[2])

    top = min(y)
    bot = max(y)
    left = min(x)
    right = max(x)
    start = (left, top)
    end = (right, bot)
    return start, end


def get_list(results_loc, img_loc):
    lm_list = []

    for id, lm in enumerate(results_loc.pose_landmarks.landmark):
        h, w, c = img_loc.shape
        x, y = int(lm.x * w), int(lm.y * h)
        if lm.visibility > 0.5:
            lm_list.append([id, x, y, lm.visibility])

    return lm_list


def fallen(start, end):
    height = end[1] - start[1]
    width = end[0] - start[0]
    fallen = False
    if height < width:
        fallen = True

    return fallen


def save_boxes(start, end):
    global boxes
    f = fallen(start, end)
    boxes.append([start, end, f])


def detect_box_fall(start, end):
    counter = 0
    fall = False

    for box in boxes:
        if box[2]:
            counter += 1

    if counter > 8:
        fall = True

    return fall


def detect_lines_fall(results_loc):
    # code von Artem
    return True


def detect_lm_fall(lm_list, img_loc):
    fall = False
    h, w, c = img_loc.shape
    under_half = 0
    for lm in lm_list:
        if lm[2] > h/2:
            under_half += 1

    if under_half/len(lm_list) > 0.8:
        fall = True
    return fall


def setup_img(img_loc, start, end, lm_list, box_fall, lines_fall, lm_fall):
    if len(lm_list) > 1:
        if box_fall:
            text = "Gefallen"
        else:
            text = "Steht noch"
        save_boxes(start, end)
        if boxes[-1][2]:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        img_loc = cv2.rectangle(img_loc, start, end, color, thickness=3)
        img_loc = cv2.putText(img_loc, text, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, color)

    return img_loc


def start_detection(results_loc, img_loc):
    global boxes
    lm_list = get_list(results_loc, img_loc)

    start, end = get_bounding_box(lm_list)
    box_fall = detect_box_fall(start, end)
    lines_fall = detect_lines_fall(results_loc)
    lm_fall = detect_lm_fall(lm_list, img)
    img_loc = setup_img(img_loc, start, end, lm_list, box_fall, lines_fall, lm_fall)

    return img_loc


if __name__ == "__main__":

    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = pose.process(imgRGB)

        if results.pose_landmarks:
            img = start_detection(results, img)

            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
