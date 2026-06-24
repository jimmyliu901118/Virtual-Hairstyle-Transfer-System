"""
代碼功能：
1. 用 dlib 人臉檢測器檢測出人脸，返回的人臉矩形框
2. 對檢測出的人臉進行關鍵點檢測並用圈進行標記
3. 將檢測出的人臉關鍵點信息寫到 txt 文本中
"""
import cv2
import dlib
import numpy as np


predictor_model = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector() # dlib 人臉檢測器
predictor = dlib.shape_predictor(predictor_model)

# cv2 讀取圖像
test_img_path = "Messi.jpg"
output_pos_info = "output_pos_info/Messi.txt"
img = cv2.imread(test_img_path)
file_handle = open(output_pos_info, 'a')

# 取灰度
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 人臉數 rects（rectangles）
rects = detector(img_gray, 0)


for i in range(len(rects)):
    landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[i]).parts()])
    for idx, point in enumerate(landmarks):
        # 68 個點的座標
        pos = (point[0, 0], point[0, 1])
        print(idx+1, pos)
        pos_info = str(point[0, 0]) + ' ' + str(point[0, 1]) + '\n'
        file_handle.write(pos_info)
        # 利用 cv2.circle 给每個特征點畫一個圈，共 68 個
        cv2.circle(img, pos, 3, color=(0, 255, 0))
        # 利用 cv2.putText 輸出 1-68
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(idx+1), pos, font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

file_handle.close()
cv2.imwrite("output/Messi_keypoints.png", img)