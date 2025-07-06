import random
import cv2
import cvzone
import time
from ultralytics import YOLO

# Load trained YOLO model
model = YOLO("weights/best.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Game state
stateResult = False
startGame = False
scores = [0, 0]  # [AI, Player]
initialTime = 0
timer = 0
playerMove = None
aiMove = None

while True:
    imgBG = cv2.imread("Resources/BG.png")

    # webcam
    success, img = cap.read()
    if not success:
        print("Webcam not found.")
        break

    img = cv2.flip(img, 1)
    imgScaled = cv2.resize(img, (0, 0), fx=0.875, fy=0.875)
    imgScaled = imgScaled[:, 80:480]

    imgBG[310:730, 780:1180] = imgScaled

    if startGame:
        if not stateResult:
            timer = time.time() - initialTime
            cv2.putText(imgBG, str(int(timer)), (605, 510),
                        cv2.FONT_HERSHEY_PLAIN, 6, (0, 0, 0), 4)

            if timer > 3:
                stateResult = True
                timer = 0

                # detect using YOLO
                results = model.predict(imgScaled, conf=0.5, verbose=False)
                playerMove = None

                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        # map class index to your numbering:
                        if cls == 0:
                            playerMove = 2  # Paper
                        elif cls == 1:
                            playerMove = 1  # Rock
                        elif cls == 2:
                            playerMove = 3  # Scissors

                # random AI move
                aiMove = random.randint(1, 3)
                ai_img = cv2.imread(f"Resources/{aiMove}.png", cv2.IMREAD_UNCHANGED)
                imgBG = cvzone.overlayPNG(imgBG, ai_img, (149, 310))

                # winner logic
                if playerMove is not None:
                    if (playerMove == 1 and aiMove == 3) or \
                       (playerMove == 2 and aiMove == 1) or \
                       (playerMove == 3 and aiMove == 2):
                        scores[1] += 1
                    elif (aiMove == 1 and playerMove == 3) or \
                         (aiMove == 2 and playerMove == 1) or \
                         (aiMove == 3 and playerMove == 2):
                        scores[0] += 1

    if stateResult:
        if playerMove is not None:
            # re-show ai image
            ai_img = cv2.imread(f"Resources/{aiMove}.png", cv2.IMREAD_UNCHANGED)
            imgBG = cvzone.overlayPNG(imgBG, ai_img, (149, 310))

            # show text
            if (playerMove == 1 and aiMove == 3) or \
               (playerMove == 2 and aiMove == 1) or \
               (playerMove == 3 and aiMove == 2):
                cv2.putText(imgBG, "Player Wins", (540, 480),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 4)
            elif (aiMove == 1 and playerMove == 3) or \
                 (aiMove == 2 and playerMove == 1) or \
                 (aiMove == 3 and playerMove == 2):
                cv2.putText(imgBG, "AI Wins", (540, 480),
                            cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 4)
            else:
                cv2.putText(imgBG, "DRAW", (540, 480),
                            cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 5)

    # scores
    cv2.putText(imgBG, str(scores[0]), (410, 260), cv2.FONT_HERSHEY_PLAIN, 4,
                (255, 255, 255), 6)
    cv2.putText(imgBG, str(scores[1]), (1112, 260), cv2.FONT_HERSHEY_PLAIN, 4,
                (255, 255, 255), 6)

    # instructions
    cv2.putText(imgBG, "For Reset -> press 'r'", (80, 720),
                cv2.FONT_HERSHEY_PLAIN, 2, (100, 100, 255), 3)
    cv2.putText(imgBG, "For Quit -> press 'q'", (790, 720),
                cv2.FONT_HERSHEY_PLAIN, 2, (100, 100, 255), 3)

    # show window
    cv2.imshow("Rock Paper Scissors", imgBG)

    key = cv2.waitKey(1)
    if key == ord('r'):
        startGame = True
        initialTime = time.time()
        stateResult = False
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


