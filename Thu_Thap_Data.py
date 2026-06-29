import cv2
import numpy as np
import os

# Tạo thư mục chứa toàn bộ ảnh cắt được
if not os.path.exists('dataset/all_crops'):
    os.makedirs('dataset/all_crops')

def collect_dataset(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return
        
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    
    frame_count = 0 

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        fg_mask = bg_subtractor.apply(frame)

        # Mặt nạ ROI
        roi_mask = np.zeros((video_h, video_w), dtype=np.uint8)
        roi_poly = np.array([[[int(video_w * 0.15), video_h], [int(video_w * 0.35), int(video_h * 0.52)], 
                              [int(video_w * 0.65), int(video_h * 0.52)], [int(video_w * 0.85), video_h]]], dtype=np.int32)
        cv2.fillPoly(roi_mask, roi_poly, 255)
        fg_mask = cv2.bitwise_and(fg_mask, roi_mask)

        # BỎ HẲN ĐIỀU KIỆN SKIP FRAME VÀ SCENE CHANGE Ở ĐÂY
        # Cứ để nó chạy và bắt tự do

        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 25)), iterations=1)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_contour = None
        max_area = 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if h == 0: continue
            
            if area > 1000 and (w/h) < 1.2 and w < (video_w * 0.2) and h < (video_h * 0.4):
                if area > max_area:
                    max_area = area
                    best_contour = contour

        if best_contour is not None:
            x, y, w, h = cv2.boundingRect(best_contour)
            
            # CỨ MỖI 5 FRAME, LƯU MỘT ẢNH (Bất kể đúng sai)
            if frame_count % 5 == 0:
                try:
                    crop_img = frame[y:y+h, x:x+w]
                    crop_img = cv2.resize(crop_img, (64, 128)) # Ép về size chuẩn để train CNN
                    cv2.imwrite(f"dataset/all_crops/frame_{frame_count}.jpg", crop_img)
                except Exception as e:
                    pass

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Thu thap Dataset...", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    print("Đã gom xong data!")

collect_dataset('badmintion.mp4')
