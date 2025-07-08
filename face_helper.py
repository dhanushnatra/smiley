import cv2


cascade_path="haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)


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
    print(
        detect_face_and_save(image_path="data/train/5__Neutral/ffhq_34.png",save_path="./ffhq_24.png")
    )