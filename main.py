import cv2
from model import predict

video_cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")



while True :
    ret,frame = video_cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    # keep only the largest detected face (if any)
    if len(faces) > 0:
        faces = [max(faces, key=lambda r: r[2]*r[3])]
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face_roi = cv2.cvtColor(frame[y:y+h,x:x+w], cv2.COLOR_BGR2RGB)
            cv2.putText(frame, predict(face_roi), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow('Webcam Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_cap.release()
cv2.destroyAllWindows()