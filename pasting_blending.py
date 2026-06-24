import cv2
import numpy as np

def paste_hairstyle(real_face_path, gan_hair_path, real_txt_path, gan_txt_path):
    """
    將 GAN 生成圖的髮型，經過幾何校正後拼接至真實人臉上
    """
    # 1. 讀取真實人臉影像與 GAN 生成的頭髮影像
    real_img = cv2.imread(real_face_path)   # 真人臉
    hair_img = cv2.imread(gan_hair_path)   # GAN 生成的頭髮圖
    
    if real_img is None or hair_img is None:
        print("[Error] 影像讀取失敗，請檢查路徑。")
        return None

    # 2. 模擬讀取 dlib 輸出的 68 點座標檔 (此處用模擬資料代替真實讀檔)
    # 在實務上，你們會解析 Messi.txt 讀入 X, Y 座標
    # 這裡我們挑選三個黃金基準點：左眼中心(36)、右眼中心(45)、鼻尖(30) 來做三角變換
    # 假設從 txt 讀出來的座標如下：
    src_tri = np.array([[180, 200], [320, 200], [250, 280]], dtype=np.float32) # GAN圖中的三點
    dst_tri = np.array([[190, 220], [310, 215], [245, 295]], dtype=np.float32) # 真人圖中的三點

    # 3. 核心幾何校正：計算仿射變換矩陣 (Affine Matrix)
    # 這個矩陣包含了「轉正角度」、「放大縮小比例」與「平移位置」
    warp_mat = cv2.getAffineTransform(src_tri, dst_tri)
    
    # 4. 對整張頭髮影像進行空間扭曲對齊 (Warping)
    # 讓頭髮的尺寸和角度完全符合真人臉的骨架
    rows, cols, _, = real_img.shape
    warped_hair = cv2.warpAffine(hair_img, warp_mat, (cols, rows), flags=cv2.INTER_CUBIC)

    # 5. 建立髮型擷取遮罩 (Hairstyle Mask)
    # 在實務上，你們會利用臉部上緣的特徵點（0~16號點）以上當作頭髮區域
    # 這裡我們模擬建立一個上半部的頭髮遮罩（255為要保留的頭髮，0為不要的臉）
    mask = np.zeros((rows, cols), dtype=np.uint8)
    # 模擬畫一個涵蓋頭髮區域的多邊形 (例如額頭上方)
    hair_poly = np.array([[0, 0], [cols, 0], [cols, int(rows*0.45)], [0, int(rows*0.45)]], dtype=np.int32)
    cv2.fillPoly(mask, [hair_poly], 255)
    
    # 讓邊緣羽化（模糊化），拼接時才不會有死硬的割裂線
    mask_blur = cv2.GaussianBlur(mask, (21, 21), 0) / 255.0
    mask_blur = np.expand_dims(mask_blur, axis=2) # 變成 3 通道比例

    # 6. 線性融合 (Linear Blending)：把校正後的頭髮與真人臉貼合
    # 公式：輸出 = 頭髮 * 遮罩權重 + 真人臉 * (1 - 遮罩權重)
    output = (warped_hair * mask_blur + real_img * (1.0 - mask_blur)).astype(np.uint8)

    return output

# =========================================================================
# 執行測試
# =========================================================================
if __name__ == "__main__":
    print("[Processing] 正在執行髮型與人臉的幾何拼接補丁...")
    
    # 建立兩個模擬的 dummy 影像來確認程式可以跑通
    real_mock = np.ones((500, 500, 3), dtype=np.uint8) * 200 # 淺灰底代表真人臉
    hair_mock = np.zeros((500, 500, 3), dtype=np.uint8)     # 黑底
    cv2.circle(hair_mock, (250, 100), 120, (100, 50, 0), -1) # 畫一個棕色圓圈代表 GAN 生成的頭髮
    
    cv2.imwrite("mock_real.jpg", real_mock)
    cv2.imwrite("mock_gan_hair.jpg", hair_mock)
    
    # 呼叫拼接功能
    result_img = paste_hairstyle("mock_real.jpg", "mock_gan_hair.jpg", "", "")
    
    if result_img is not None:
        cv2.imwrite("final_virtual_tryon.jpg", result_img)
        print("[SUCCESS] 補丁程式執行成功！最終拼接成果已儲存至 'final_virtual_tryon.jpg'。")