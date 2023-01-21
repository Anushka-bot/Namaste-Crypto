import time
from flask import Flask, render_template, stream_with_context, request, Response
import mediapipe as mp
import cv2
import os
import json
from PIL import Image
from numpy import asarray
from matplotlib import pyplot as plt
import numpy as np
from playsound import playsound
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify
from playsound import playsound
from threading import Thread


mp_drawing = mp.solutions.drawing_utils  # gives all the drwing utilities
mp_pose = mp.solutions.pose  # grab the pose model out of all the models available
mp_drawing_styles = mp.solutions.drawing_styles

# folder = path = os.getcwd() + "\\yoga_poses\\ExercisePoses"
frameX = 0
frameY = cv2.imencode(".jpg", cv2.imread("pic4.jpg"))[1].tobytes()
amount = 0
flag = 0
app = Flask(__name__)


def gen():
    global flag
    if flag != 1:
        previous_time = 0
        # creating our model to draw landmarks
        mpDraw = mp.solutions.drawing_utils
        # creating our model to detected our pose
        my_pose = mp.solutions.pose
        pose = my_pose.Pose()
        global frameY
        global amount
        """Video streaming generator function."""
        # cap = cv2.VideoCapture("./test.mp4")
        def test(b):
            playsound("mybeep4.mp3")

        def landmark_list(res):
            if res.pose_landmarks is None:
                return []
            return list(res.pose_landmarks.landmark)

        def get_coord(l):
            coord = []
            for i in l:
                x = i.x
                y = i.y
                coord.append([x, y])
            return coord

        def check_matching(res1, res2, threshold):
            l1 = landmark_list(res1)
            if not l1:
                return False
            coord1 = get_coord(l1)
            primary = np.array(coord1)

            # res2 = get_skeleton(im2)
            l2 = landmark_list(res2)
            coord2 = get_coord(l2)
            secondary = np.array(coord2)

            n = primary.shape[0]
            pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
            unpad = lambda x: x[:, :-1]
            X = pad(primary)
            Y = pad(secondary)

            # Solve the least squares problem X * A = Y
            # to find our transformation matrix A
            A, res, rank, s = np.linalg.lstsq(X, Y)

            transform = lambda x: unpad(np.dot(pad(x), A))
            m = np.abs(secondary - transform(primary)).max()
            if m <= threshold:
                return True
            return False

        while True:
            poses = [15.0, 90.0, 165.0, 316.0, 465.0, 564.0, -1]
            cap = cv2.VideoCapture(0)
            cap2 = cv2.VideoCapture("side2.mp4")

            frame2 = 0
            fm = 0
            imagex = 0
            resultx = 0
            cnt = 0
            ## Setup mediapipe instance
            with mp_pose.Pose(
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            ) as pose:
                while cap.isOpened():
                    try:
                        ret, frame = cap.read()
                        # Recolor image to RGB
                        # Recolor image to RGB
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False

                        # Make detection
                        results = pose.process(image)

                        # Recolor back to BGR
                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                        # Render detections
                        mp_drawing.draw_landmarks(
                            image,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(
                                color=(146, 173, 254), thickness=2, circle_radius=2
                            ),
                            mp_drawing.DrawingSpec(
                                color=(109, 138, 227), thickness=2, circle_radius=2
                            ),
                        )
                        # cv2.imshow("Mediapipe Feed", image)
                        if fm != poses[cnt]:
                            fm = cap2.get(cv2.CAP_PROP_POS_FRAMES)
                            ret, frame2 = cap2.read()
                            # img_me = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                            # img_me.flags.writeable = False
                            # results_me = pose.process(img_me)
                            # img_me.flags.writeable = True
                            # img_me = cv2.cvtColor(img_me, cv2.COLOR_RGB2BGR)
                            # mp_drawing.draw_landmarks(
                            #     img_me,
                            #     results_me.pose_landmarks,
                            #     mp_pose.POSE_CONNECTIONS,
                            #     mp_drawing.DrawingSpec(
                            #         color=(227,138,109), thickness=2, circle_radius=2
                            #     ),
                            #     mp_drawing.DrawingSpec(
                            #         color=(245, 66, 230), thickness=2, circle_radius=2
                            #     ),
                            # )
                            cv2.rectangle(
                                frame2,
                                (760, 960),
                                (1100, 1100),
                                (215, 156, 248),
                                -1,
                            )
                            imagex = frame2
                            frameX = frame2

                            # cv2.imshow("Media Feed", frame2)
                            # frame4 = cv2.imencode(".jpg", frame2)[1].tobytes()
                            # yield (
                            #     b"--frame4\r\n"
                            #     b"Content-Type: image/jpeg\r\n\r\n" + frame4 + b"\r\n"
                            # )
                        else:
                            image2 = imagex
                            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                            image2.flags.writeable = False
                            # Make detection
                            if resultx == 0:
                                results2 = pose.process(image2)
                                resultx = results2
                                # Recolor back to BGR
                            image2.flags.writeable = True
                            image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
                            cv2.putText(
                                image2,
                                "PLEASE MATCH THE",
                                (773, 1000),
                                cv2.FONT_HERSHEY_DUPLEX,
                                0.8,
                                (8, 8, 9),
                                2,
                                cv2.LINE_AA,
                            )
                            cv2.putText(
                                image2,
                                "YOGA POSITION",
                                (773, 1030),
                                cv2.FONT_HERSHEY_DUPLEX,
                                0.8,
                                (8, 8, 9),
                                2,
                                cv2.LINE_AA,
                            )
                            cv2.putText(
                                image2,
                                "SHOWN HERE",
                                (773, 1060),
                                cv2.FONT_HERSHEY_DUPLEX,
                                0.8,
                                (8, 8, 9),
                                2,
                                cv2.LINE_AA,
                            )
                            # mp_drawing.draw_landmarks(
                            #     image2,
                            #     resultx.pose_landmarks,
                            #     mp_pose.POSE_CONNECTIONS,
                            #     mp_drawing.DrawingSpec(
                            #         color=(227,138,109), thickness=2, circle_radius=2
                            #     ),
                            #     mp_drawing.DrawingSpec(
                            #         color=(245, 66, 230), thickness=2, circle_radius=2
                            #     ),
                            # )
                            frameX = image2
                            # cv2.imshow("Media Feed", image2)
                            threshold = 0.15
                            if check_matching(results, resultx, threshold):
                                print("matched.........")
                                cnt += 1
                                amount += 10
                                Thread(target=test, args=(5,)).start()
                                # playsound("beep.wav")
                        frame3 = cv2.imencode(".jpg", image)[1].tobytes()
                        frame4 = cv2.imencode(".jpg", frameX)[1].tobytes()
                        frameY = frame4
                        if cv2.waitKey(10) & 0xFF == ord("q"):
                            break
                            # cv2.imshow("Pose detection", img)
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + frame3 + b"\r\n"
                        )
                        key = cv2.waitKey(20)
                        if key == 27:
                            break

                    except:
                        break
            flag = 1
            print(flag)
            cap.release()
            cv2.destroyAllWindows()


@app.route("/video_feed")
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(
        stream_with_context(gen()), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/stream")
def streamed_response():
    def generate():
        while True:
            yield (
                b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frameY + b"\r\n"
            )

    return Response(
        stream_with_context(generate()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/rewards")
def streamed_res():
    are = str(amount)
    return {"class": amount}


# def curls():
#     global amount

#     def calculate_angle(a, b, c):
#         a = np.array(a)
#         b = np.array(b)
#         c = np.array(c)  # here c[0]is x of the point and c[1] is y
#         radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
#             a[1] - b[1], a[0] - b[0]
#         )  # we shift the center to b and calculate the angle by angle(c-b)-angle(a-b) in radians
#         angle = np.abs(radians * 180.0 / np.pi)

#         if angle > 180.0:
#             angle = 360.0 - angle
#         return angle

#     def test(b):
#         playsound("beep.wav")

#     cap = cv2.VideoCapture("side.mp4")
#     counter = 0
#     stage = None
#     with mp_pose.Pose(
#         min_detection_confidence=0.5, min_tracking_confidence=0.5
#     ) as pose:
#         while cap.isOpened():
#             ret, frame = cap.read()

#             # Recolor image to RGB
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image.flags.writeable = False

#             # Make detection
#             results = pose.process(image)

#             # Recolor back to BGR
#             image.flags.writeable = True
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#             # Extract landmarks
#             try:
#                 landmarks = results.pose_landmarks.landmark

#                 # Get coordinates
#                 shoulder = [
#                     landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
#                     landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
#                 ]
#                 elbow = [
#                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
#                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
#                 ]
#                 wrist = [
#                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
#                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
#                 ]

#                 # Calculate angle
#                 angle = calculate_angle(shoulder, elbow, wrist)

#                 # Visualize angle
#                 cv2.putText(
#                     image,
#                     str(angle),
#                     tuple(np.multiply(elbow, [640, 480]).astype(int)),
#                     cv2.FONT_HERSHEY_DUPLEX,
#                     0.5,
#                     (255, 255, 255),
#                     2,
#                     cv2.LINE_AA,
#                 )

#                 # Curl counter logic
#                 if angle > 135:
#                     stage = "down"
#                 if angle < 49 and stage == "down":
#                     stage = "up"
#                     counter += 1
#                     amount += 1
#                     Thread(target=test, args=(5,)).start()

#                     # print(counter)

#             except:
#                 pass

#             # Render curl counter
#             # Setup status box
#             cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)

#             # Rep data
#             cv2.putText(
#                 image,
#                 "REPS",
#                 (15, 12),
#                 cv2.FONT_HERSHEY_DUPLEX,
#                 0.5,
#                 (143,140,145),
#                 1,
#                 cv2.LINE_AA,
#             )
#             cv2.putText(
#                 image,
#                 str(counter),
#                 (10, 60),
#                 cv2.FONT_HERSHEY_DUPLEX,
#                 2,
#                 (255, 255, 255),
#                 2,
#                 cv2.LINE_AA,
#             )

#             # Stage data
#             cv2.putText(
#                 image,
#                 "STAGE",
#                 (65, 12),
#                 cv2.FONT_HERSHEY_DUPLEX,
#                 0.5,
#                 (143,140,145),
#                 1,
#                 cv2.LINE_AA,
#             )
#             cv2.putText(
#                 image,
#                 stage,
#                 (60, 60),
#                 cv2.FONT_HERSHEY_DUPLEX,
#                 2,
#                 (255, 255, 255),
#                 2,
#                 cv2.LINE_AA,
#             )

#             # Render detections
#             mp_drawing.draw_landmarks(
#                 image,
#                 results.pose_landmarks,
#                 mp_pose.POSE_CONNECTIONS,
#                 landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
#                 # mp_drawing.DrawingSpec(
#                 #     color=(227,138,109), thickness=2, circle_radius=2
#                 # ),
#                 # mp_drawing.DrawingSpec(
#                 #     color=(245, 66, 230), thickness=2, circle_radius=2
#                 # ),
#             )

#             cv2.imshow("Mediapipe Feed", image)
#             frame3 = cv2.imencode(".jpg", image)[1].tobytes()
#
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()


# @app.route("/gym")
# def mygym():
#     return Response(
#         stream_with_context(curls()),
#         mimetype="multipart/x-mixed-replace; boundary=frame",
#     )


if __name__ == "__main__":
    app.run(debug=True)
