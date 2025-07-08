import cv2
from use_model import predict
import time
import asyncio
import numpy as np
cascade_path="haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)

# Emotion to emoji mapping
emotion_emojis = {
    "Anger": "😠",
    "Contempt": "😤", 
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😊",
    "Neutral": "😐",
    "Sad": "😢",
    "Surprise": "😲"
}

















async def open_cam_and_detect_face(save_path):
    cap = cv2.VideoCapture(0)
    last_prediction_time = 0
    current_emotion = None
    prediction_interval = 2 # Predict every 6 seconds
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # print(f"Detected {len(faces)} faces")

        current_time = time.time()
        
        # Only predict emotion every 6 seconds
        if len(faces) > 0 and (current_time - last_prediction_time) >= prediction_interval:
            (x, y, w, h) = faces[0]
            face_img = frame[y:y+h, x:x+w]
            current_emotion = predict(face_img)
            last_prediction_time = current_time
            
        # Display the last predicted emotion (even if not predicting this frame)
        if current_emotion and len(faces) > 0:
            emotion_text = current_emotion['emotion']
            emoji = emotion_emojis.get(emotion_text, "😐")  # Default to neutral if emotion not found
            print(f"Detected Emotion: {emotion_text} {emoji}")
            
            # Display emotion and emoji on the frame
            (x, y, w, h) = faces[0]
            # Put text above the rectangle
            cv2.putText(frame,f"{emoji}", (x, y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            cv2.imshow('Face Detection', frame)
        else:
            print("No faces detected or no emotion predicted yet.")
        key = cv2.waitKey(1)
        if key == ord('s') and len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_img = frame[y:y+h, x:x+w]
            cv2.imwrite(save_path, face_img)
            print(f"Face saved to {save_path}")
        if key == ord('q'):
            print("face array",type(faces[0]))
            break
    cap.release()
    cv2.destroyAllWindows()


def detect_face(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_img = image[y:y+h, x:x+w]
        emotion = predict(face_img)
        emoji = emotion_emojis.get(emotion['emotion'], "😐")  # Default
        return {
            "emotion": emotion['emotion'],
            "emoji": emoji,
        }
    else:
        return None



def detect_from_bytes(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_img = image[y:y+h, x:x+w]
            emotion = predict(face_img)
            emoji = emotion_emojis.get(emotion['emotion'], "😐")  # Default
            return {
                "face_dimensions": [x, y, w, h],
                "emotion": str(emotion['emotion']),
            }
        else:
            return None
    except Exception as e:
        print(f"Error processing image bytes: {e}")
        return None






def detect_face_and_save(image_path,save_path, ):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_img = image[y:y+h, x:x+w]
        cv2.imwrite(save_path, face_img)
    else:
        cv2.imwrite(save_path,image)
    return f"face saved to path {save_path} from {image_path}"


if __name__=="__main__":
    asyncio.run(open_cam_and_detect_face("test_image.jpg"))