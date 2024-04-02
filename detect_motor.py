import cv2
import torch


def draw_line(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, drawing
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that drawing is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the drawing operation is finished
        refPt.append((x, y))
        drawing = False
        # draw a rectangle around the region of interest
        cv2.line(first_frame, refPt[0], refPt[1], (0, 0, 255), 2)


def calculate_x(y):
    global refPt
    if refPt[0][0] < refPt[1][0]:
        return refPt[1][0]-(abs(refPt[0][0]-refPt[1][0])*abs(y-refPt[1][1])/abs(refPt[0][1]-refPt[1][1]))
    else:
        return (abs(refPt[0][0]-refPt[1][0])*abs(y-refPt[1][1])/abs(refPt[0][1]-refPt[1][1]))+refPt[1][0]


repo_yolov7_path = 'yolov7'
weights_path = 'Do_an/best.pt'
video_path = 'VID_20221207_120509.mp4'

# model = torch.hub.load(repo_yolov5_path, 'custom', path=weights_path, source='local')

model = torch.hub.load("yolov7", "custom", 'yolov7.pt', source='local', trust_repo=True, force_reload=True)
model.conf = 0.3


b = model.names[9] = 'motor'
color = (0, 0, 255)
offset = 2
counter = 0
refPt = []
drawing = True


cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

result = cv2.VideoWriter('output.mp4',
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         23, size)


success = True
success, first_frame = cap.read()
cv2.imshow('Frame', first_frame)
cv2.setMouseCallback("Frame", draw_line)

while drawing:
    # display the image and wait for a keypress
    cv2.imshow("Frame", first_frame)

    key = cv2.waitKey(1) & 0xFF
    # if the 'c' key is pressed, break from the loop
    if key == ord("c"):
        break


while success:
    success, frame = cap.read()

    results = model(frame)

    h, w, depth = frame.shape
    # vẽ đường thẳng để làm mốc đếm
    cv2.line(frame, refPt[0], refPt[1], (0, 0, 255), 2)

    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        d = (row['class'])
        if d == 9:  # chỉ lấy class 3 = car
            # vẽ bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # vẽ tâm bounding box
            rectx1, recty1 = ((x1+x2)/2, (y1+y2)/2)
            rectcenter = int(rectx1), int(recty1)
            cx = rectcenter[0]
            cy = rectcenter[1]
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0))
            # ghi tên class
            # cv2.putText(frame, str(b), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), thickness=2)

            if (cy <= refPt[1][1] and cy >= refPt[0][1]):
                x_line = calculate_x(cy)
                if cx < (x_line+offset) and cx > (x_line-offset):
                    counter += 1
                    cv2.line(frame, refPt[0], refPt[1], (0, 255, 0), 2)

    # ghi số lượng đếm được
    cv2.putText(frame, str("number of motor: {}".format(counter)), (50, h-50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), thickness=4)

    cv2.imshow('Frame', frame)

    result.write(frame)

    # Press S on keyboard
    # to stop the process
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
result.release()
cv2.destroyAllWindows()
print("The video was successfully saved")
