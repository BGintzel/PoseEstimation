import cv2
import mediapipe as mp


def get_boundingbox(list):
    x = []
    y = []
    for lm in list:
        x.append(lm[1])
        y.append(lm[2])

    top = min(y)
    bot = max(y)
    left = min(x)
    right = max(x)
    start = (left, top)
    end = (right, bot)
    return start, end


def get_list(results, img):
    lm_list = []

    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
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
    f = fallen(start, end)
    boxes.append([start, end, f])


def detect_fall(start, end):
    counter = 0
    fall = False

    for box in boxes[-9:]:
        if box[2]:
            counter += 1

    if counter > 8:
        fall = True

    return fall


def start_detection(results, img):
    lm_list = get_list(results, img)

    start, end = get_boundingbox(lm_list)
    if len(lm_list) > 1:
        if detect_fall(start, end):
            text = "Gefallen"
        else:
            text = "Steht noch"
        save_boxes(start, end)
        if boxes[-1][2]:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        img = cv2.rectangle(img, start, end, color, thickness=3)
        img = cv2.putText(img, text, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, color)

        return img


if __name__ == "__main__":
    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    cap = cv2.VideoCapture(0)

    boxes = []
    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = pose.process(imgRGB)

        if results.pose_landmarks:
            img = start_detection(results, img)

            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
