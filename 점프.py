import cv2
import mediapipe as mp
import time as time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def detectPoseOnWebcam(pose):
    cap = cv2.VideoCapture(0)
    prev_left_ankle_y = 0 # 왼쪽 발
    prev_right_ankle_y = 0 # 오른쪽 발
    test = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)
        height, width, _ = frame.shape

        if results.pose_landmarks: # 포즈 랜드마크가 존재하는가
            mp_drawing.draw_landmarks(image=frame, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS) # 시각화

            curr_left_ankle_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * height # 왼쪽 발목 y좌표 계산
            curr_right_ankle_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * height # 오른쪽 발목 y좌표 계산

            diff_left = curr_left_ankle_y - prev_left_ankle_y # 이전 프레임과 비교하기
            diff_right = curr_right_ankle_y - prev_right_ankle_y # 이전 프레임과 비교하기

            if diff_left > 25 or diff_right > 25: # 25 이상이면 + 1 해서 출력
                test = test + 1  
                print(test)

            prev_left_ankle_y = curr_left_ankle_y
            prev_right_ankle_y = curr_right_ankle_y

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
detectPoseOnWebcam(pose)
