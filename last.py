import numpy as np
import cv2
import keras


model = keras.models.load_model("trained_model")

bg = None
aWeight = 0.5
nr_frames = 0
nr_imgs = 0
top, bottom, right, left = 100, 300, 350, 550


def run_avg(frame, aWeight):
    global bg

    if bg is None:  # first
        bg = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, bg, aWeight)


def segment_hand(frame, threshold=25):
    global bg

    diff = cv2.absdiff(frame, bg.astype("uint8"))

    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    cnts, _ = cv2.findContours(
        thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(cnts) == 0:
        return None
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


cam = cv2.VideoCapture(0)


while True:
    check, frame = cam.read()

    frame = cv2.flip(frame, 1)

    roi = frame[top:bottom, right:left]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)

    if nr_frames < 60:
        run_avg(gray, aWeight)

    else:
        hand = segment_hand(gray)
        if hand is not None:
            thresholded, segm = hand

            # cv2.drawContours(frame, [segm + (right, top)], -1, (255, 0, 127), 1)
            cv2.imshow("Black and White", thresholded)

            thresholded = cv2.resize(thresholded, (64, 64))
            thresholded = np.reshape(
                thresholded, (1, thresholded.shape[0], thresholded.shape[1], 1)
            )

            class_names = ["Saluuut", "Te iubesc", "Daaa"]

            predictions = model.predict(thresholded)
            label = class_names[np.argmax(predictions)]
            cv2.putText(
                frame,
                str(label),
                (400, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (102, 0, 204),
                2,
            )
        else:
            cv2.putText(
                frame,
                "Unde???",
                (380, 380),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (102, 0, 204),
                2,
            )

    cv2.rectangle(frame, (left, top), (right, bottom), (255, 128, 0), 3)

    nr_frames += 1

    cv2.imshow("Main", frame)

    if cv2.waitKey(1) == ord("p"):
        break


cv2.destroyAllWindows()
cam.release()
