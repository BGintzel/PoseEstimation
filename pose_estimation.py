import cv2
import mediapipe as mp

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture(0)


def getBoundingbox(list):
    x=[]
    y=[]
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


def getList(results, img):
    lm_list = []

    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        x, y = int(lm.x * w), int(lm.y * h)
        if lm.visibility>0.5:
            lm_list.append([id, x, y, lm.visibility])

    return lm_list

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = pose.process(imgRGB)
    #print(results.pose_landmarks.landmark)
    if results.pose_landmarks:
        list = getList(results, img)
        print(list)
        start, end = getBoundingbox(list)
        if len(list) > 15:
            img = cv2.rectangle(img, start, end, color=(0, 0, 0),thickness=3)
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    cv2.imshow("Image", img)
    cv2.waitKey(1)



