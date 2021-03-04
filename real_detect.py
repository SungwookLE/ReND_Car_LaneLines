import cv2
from lane_detect import process_image

#REAL TEST

""" '21.3/4
NOTE: Under ubuntu 20.04, 
Gtk-Message: 23:16:07.192: Failed to load module "atk-bridge"
Gtk-Message: 23:16:07.192: Failed to load module "appmenu-gtk-module"
Gtk-Message: 23:16:07.194: Failed to load module "canberra-gtk-module"

QObject::moveToThread: Current thread (0x5613cb136390) is not the object's thread (0x7f99cfbbb740).
Cannot move to target thread (0x5613cb136390)
...
"""

cap = cv2.VideoCapture(0)
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while(cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.resize(frame, dsize=(960,540))
        print(frame.shape)
        out=process_image(frame,'real')
        cv2.imshow('out',out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()    
