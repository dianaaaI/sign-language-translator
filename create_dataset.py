import cv2
import os

bg = None
aWeight = 0.5
nr_frames = 0
nr_imgs = 0
top, bottom, right, left = 100, 300, 350, 550

dir0 = "test"

try:
    os.mkdir(dir0)
except FileExistsError:
    print("exists")

dir1 = os.path.join(dir0, "Yes")

try:
    os.mkdir(dir1)
except FileExistsError:
    print("exists")


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
# cam = cv2.VideoCapture('http://192.168.0.23:8080/video')

while True:
    check, frame = cam.read()

    frame = cv2.flip(frame, 1)
    roi = frame[top:bottom, right:left]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)


    if nr_frames < 60:
        run_avg(gray, aWeight)

    elif nr_frames <= 200:
        hand = segment_hand(gray)

        cv2.putText(
            frame,
            "Hopa sus",
            (380, 380),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 128, 255),
            2,
        )

        if hand is not None:
            thresholded, segm = hand
            cv2.drawContours(frame, [segm + (right, top)], -1, (255, 0, 127), 1)

            cv2.imshow("Black and White", thresholded)

    else:
        hand = segment_hand(gray)
        if hand is not None:
            thresholded, segm = hand

            cv2.drawContours(frame, [segm + (right, top)], -1, (255, 0, 127), 1)

            cv2.putText(
                frame,
                str(nr_imgs),
                (400, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (102, 0, 204),
                2,
            )

            cv2.imshow("Black and White", thresholded)
            if nr_imgs <= 200:
                img_path = os.path.join(dir1, f"{nr_imgs}.jpg")
                cv2.imwrite(img_path, thresholded)
                nr_imgs += 1
            else:
                break

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
