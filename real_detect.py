import cv2
from lane_detect import process_image

#REAL TEST

cap = cv2.VideoCapture(0)
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while(cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.resize(frame, dsize=(960,540))
        out=process_image(frame)
        cv2.imshow('out',out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()    
