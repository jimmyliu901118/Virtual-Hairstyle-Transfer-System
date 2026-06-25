import cv2
import dlib
import numpy as np
import os

predictor_model = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor(predictor_model)

# 🎯 【手動修改區】你想偵測哪張圖，就改這兩個名字！
test_img_path = "Messi.jpg"           # 換成你想跑的圖片名稱 (例如 "hair_50.jpg")
output_pos_info = "Messi.txt"         # 換成你想輸出的 txt 名稱 (例如 "hair_50.txt")

img = cv2.imread(test_img_path)
if img is None:
    print(f"[Error] 找不到圖片: {test_img_path}")
    exit()

# 確保直接寫在當前資料夾，不另外建資料夾以免報錯
file_handle = open(output_pos_info, 'w') # 改用 'w'，每次重跑就覆蓋舊點，才不會疊加

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rects = detector(img_gray, 0)

if len(rects) == 0:
    print(f"[Warning] {test_img_path} 沒偵測到人臉！")

for i in range(len(rects)):
    landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        pos_info = str(point[0, 0]) + ' ' + str(point[0, 1]) + '\n'
        file_handle.write(pos_info)
        cv2.circle(img, pos, 3, color=(0, 255, 0))

file_handle.close()
print(f"[SUCCESS] {test_img_path} 偵測完成，座標已寫入 {output_pos_info}")
