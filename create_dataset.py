import os
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
            os.system(f"cp dataset/{ty}/images/{label[:-3]}jpg data/{ty}/{cl}__{emotions[cl]}/")
        else:
            os.system(f"cp dataset/{ty}/images/{label[:-3]}png data/{ty}/{cl}__{emotions[cl]}/")

print("Dataset (Train , Test , Valid ) created successfully in the 'data' directory.")