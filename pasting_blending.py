import cv2
import numpy as np

def load_landmarks_from_txt(txt_path):
    points = []
    with open(txt_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            x, y = map(int, line.strip().split())
            points.append([x, y])
    return np.array(points, dtype=np.int32)

def paste_hair_with_true_translation(real_face_path, gan_hair_path, real_txt_path, gan_txt_path):
    real_img = cv2.imread(real_face_path)   
    hair_img = cv2.imread(gan_hair_path)   
    
    if real_img is None or hair_img is None:
        print("[Error] 影像讀取失敗")
        return None

    real_pts = load_landmarks_from_txt(real_txt_path)
    gan_pts = load_landmarks_from_txt(gan_txt_path)

    # 1. 在女生的「原圖坐標系」上設計拱門遮罩
    h_rows, h_cols, _ = hair_img.shape
    hair_origin_mask = np.zeros((h_rows, h_cols), dtype=np.uint8)
    
    # 眉毛以上全部允許保留
    gan_brows_y = gan_pts[17:27, 1]
    highest_brow_y = np.min(gan_brows_y)
    hair_origin_mask[0:highest_brow_y, :] = 255
    
    # 畫出女生的額頭半圓並在遮罩上扣除
    left_brow = gan_pts[17]
    right_brow = gan_pts[26]
    center_x = int((left_brow[0] + right_brow[0]) / 2)
    center_y = highest_brow_y
    
    axes_x = int(abs(right_brow[0] - left_brow[0]) * 0.5) 
    axes_y = int(axes_x * 0.45) # 這是我們切掉的半圓形垂直半徑
    cv2.ellipse(hair_origin_mask, (center_x, center_y), (axes_x, axes_y), 0, 180, 360, 0, -1)
    
    # 🎯 2. 修正核心：只平移影像與遮罩，基準點（gan_pts）絕對不加 axes_y！
    # 這樣對齊時，才會強行把下墜後的頭髮往下壓，補滿梅西的額頭空缺
    M_translate = np.float32([[1, 0, 0], [0, 1, axes_y]])
    hair_img_shifted = cv2.warpAffine(hair_img, M_translate, (h_cols, h_rows))
    hair_mask_shifted = cv2.warpAffine(hair_origin_mask, M_translate, (h_cols, h_rows))

    # 3. 幾何對齊（拿原本未被干擾的五官特徵點去對齊梅西）
    src_tri = np.array([gan_pts[36],  gan_pts[45],  gan_pts[30]], dtype=np.float32) 
    dst_tri = np.array([real_pts[36], real_pts[45], real_pts[30]], dtype=np.float32) 
    warp_mat = cv2.getAffineTransform(src_tri, dst_tri)
    
    rows, cols, _ = real_img.shape
    warped_hair = cv2.warpAffine(hair_img_shifted, warp_mat, (cols, rows), flags=cv2.INTER_CUBIC)
    warped_mask = cv2.warpAffine(hair_mask_shifted, warp_mat, (cols, rows), flags=cv2.INTER_NEAREST)
    
    # 邊緣細緻羽化
    mask_blur = cv2.GaussianBlur(warped_mask, (11, 11), 0) / 255.0
    mask_blur = np.expand_dims(mask_blur, axis=2) 

    # 4. 最終融合
    output = (warped_hair * mask_blur + real_img * (1.0 - mask_blur)).astype(np.uint8)
    return output

# ==========================================
# 🎯 執行
# ==========================================
REAL_FACE = "Messi.jpg"
REAL_TXT  = "Messi.txt"
GAN_HAIR  = "hair_50.jpg"    
GAN_TXT   = "hair_50.txt"    

result = paste_hair_with_true_translation(REAL_FACE, GAN_HAIR, REAL_TXT, GAN_TXT)

if result is not None:
    cv2.imwrite("final_virtual_tryon.jpg", result)
    print("[SUCCESS] 真正的平移補位版完成！請刷新查看成果。")
