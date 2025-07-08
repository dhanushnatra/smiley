import os
from face_helper import detect_face_and_save
emotions = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
  ]
for ty in ["train", "test","valid"]:
    try:
        if not os.path.exists("data"):
            os.mkdir("data")
        if not os.path.exists(f"data/{ty}"):
            os.mkdir(f"data/{ty}")
        for i in range(8):     
            os.mkdir(f"data/{ty}/{i}__{emotions[i]}")
    except Exception as e:
        print(f"error making classes {e}")


    for label in os.listdir(f"dataset/{ty}/labels"):
        with open(f"dataset/{ty}/labels/{label}", "r") as labelFile:
            cl = int(labelFile.read().split(" ")[0])
        
        if os.path.exists(f"dataset/{ty}/images/{label[:-3]}jpg"):
            detect_face_and_save(image_path=f"dataset/{ty}/images/{label[:-3]}jpg",save_path=f"data/{ty}/{cl}__{emotions[cl]}/{label[:-3]}jpg")
        else:
            detect_face_and_save(image_path=f"dataset/{ty}/images/{label[:-3]}png",save_path=f"data/{ty}/{cl}__{emotions[cl]}/{label[:-3]}png")

print("Dataset (Train , Test , Valid ) created successfully in the 'data' directory.")