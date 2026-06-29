import cv2
import numpy as np
from tensorflow.keras.models import load_model

# =================================================================
# HÀM VẼ SÂN CẦU LÔNG CHUẨN TỶ LỆ TRÊN BẢN ĐỒ 2D
# =================================================================
def draw_badminton_court(img):
    color = (255, 255, 255) # Màu trắng vạch kẻ
    thick = 2 # Độ dày vạch
    
    # 1. Viền ngoài cùng của sân
    cv2.rectangle(img, (0, 0), (305, 670), color, thick)
    
    # 2. Vạch lưới (chia đôi sân)
    cv2.line(img, (0, 335), (305, 335), color, thick)
    
    # 3. Đường biên dọc đánh đơn (cách mép 0.46m)
    cv2.line(img, (23, 0), (23, 670), color, thick)
    cv2.line(img, (282, 0), (282, 670), color, thick)
    
    # 4. Vạch phát cầu gần (cách lưới 1.98m)
    cv2.line(img, (0, 236), (305, 236), color, thick)
    cv2.line(img, (0, 434), (305, 434), color, thick)
    
    # 5. Vạch phát cầu xa đánh đôi (cách biên ngang 0.76m)
    cv2.line(img, (0, 38), (305, 38), color, thick)
    cv2.line(img, (0, 632), (305, 632), color, thick)
    
    # 6. Vạch chia giữa sân (kéo từ vạch phát cầu gần đến hết sân)
    cv2.line(img, (152, 0), (152, 236), color, thick)
    cv2.line(img, (152, 434), (152, 670), color, thick)


def run_ultimate_tracker(video_path, model_path):
    # 1. NẠP MÔ HÌNH AI ĐÃ HUẤN LUYỆN
    print("Đang nạp mô hình AI...")
    model = load_model(model_path)
    print("Nạp thành công! Bắt đầu xử lý video...")

    # 2. KHỞI TẠO VIDEO VÀ BẢN ĐỒ 2D
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return
        
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Ma trận Perspective
    base_pts = np.array([[644, 377], [1271, 377], [1483, 1029], [420, 1029]], dtype=np.float32)
    scale_x = video_w / 1920.0
    scale_y = video_h / 1080.0
    src_pts = np.copy(base_pts)
    src_pts[:, 0] *= scale_x
    src_pts[:, 1] *= scale_y
    
    dst_width, dst_height = 305, 670
    dst_pts = np.array([[0, 0], [dst_width, 0], [dst_width, dst_height], [0, dst_height]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Khởi tạo sân và vẽ vạch chuẩn
    court_2d = np.zeros((dst_height, dst_width, 3), dtype=np.uint8)
    court_2d[:] = (80, 180, 80)
    draw_badminton_court(court_2d)
    
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    
    trajectory_points = [] 
    court_2d_clean = court_2d.copy() 

    # Các biến tối ưu tốc độ
    frame_count = 0
    is_valid_player = False

    # 3. VÒNG LẶP XỬ LÝ
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        fg_mask = bg_subtractor.apply(frame)

        # BỘ LỌC 1: KHIÊN CHỐNG ZOOM CAMERA
        if cv2.countNonZero(fg_mask) > (video_w * video_h * 0.1):
            cv2.imshow("Video goc (AI Tracking)", frame)
            cv2.imshow("Ban do 2D (AI Trajectory)", court_2d)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue 

        # Trải mặt nạ ROI
        roi_mask = np.zeros((video_h, video_w), dtype=np.uint8)
        roi_poly = np.array([[[int(video_w * 0.15), video_h], [int(video_w * 0.35), int(video_h * 0.52)], 
                              [int(video_w * 0.65), int(video_h * 0.52)], [int(video_w * 0.85), video_h]]], dtype=np.int32)
        cv2.fillPoly(roi_mask, roi_poly, 255)
        fg_mask = cv2.bitwise_and(fg_mask, roi_mask)

        # Xử lý hình thái học
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 25)), iterations=1)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        max_area = 0

        # Lọc thô bằng Kỷ luật Vật lý
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if h == 0: continue
            
            aspect_ratio = float(w) / h
            foot_y = int(y + h)
            
            # Chặn đứng tay áo (aspect_ratio < 0.85) và ép chân chạm đất (foot_y > 55%)
            if area > 1000 and aspect_ratio < 0.85 and w < (video_w * 0.2) and h < (video_h * 0.4) and foot_y > int(video_h * 0.55):
                if area > max_area:
                    max_area = area
                    best_contour = contour

        # BƯỚC KIỂM DUYỆT CỦA AI VÀ VẼ QUỸ ĐẠO
        if best_contour is not None:
            x, y, w, h = cv2.boundingRect(best_contour)
            foot_y = int(y + h)
            foot_x = int(x + w / 2)
            
            # Tối ưu tốc độ: Chỉ gọi AI kiểm tra 3 frame 1 lần
            if frame_count % 3 == 0:
                crop_img = frame[y:y+h, x:x+w]
                try:
                    ai_input = cv2.resize(crop_img, (64, 128))
                    ai_input = np.expand_dims(ai_input, axis=0) / 255.0
                    prediction = model.predict(ai_input, verbose=0)[0][0]
                    # Ép tự tin 95%
                    is_valid_player = (prediction > 0.95)
                except Exception:
                    pass 

            # Chỉ vẽ nếu AI cho phép
            if is_valid_player:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (foot_x, foot_y), 5, (0, 0, 255), -1)

                pts = np.array([[[foot_x, foot_y]]], dtype=np.float32)
                warped_pt = cv2.perspectiveTransform(pts, matrix)
                map_x, map_y = int(warped_pt[0][0][0]), int(warped_pt[0][0][1])
                
                if 0 <= map_x <= dst_width and 0 <= map_y <= dst_height:
                    cv2.circle(court_2d, (map_x, map_y), 3, (0, 0, 255), -1)
                    trajectory_points.append((map_x, map_y))

        cv2.imshow("Video goc (AI Tracking)", frame)
        half_court_live = court_2d[335:670, 0:dst_width]
        cv2.imshow("Ban do 2D (AI Trajectory)", half_court_live)

        # Chạy max tốc độ
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cap.release()
    
    # =================================================================
    # BƯỚC CUỐI: TẠO HEATMAP NỬA SÂN - CÓ ĐỘ TRONG SUỐT
    # =================================================================
    print("Đang tổng hợp Heatmap...")
    cv2.destroyAllWindows()
    
    heatmap_layer = np.zeros((dst_height, dst_width), dtype=np.float32)

    # Cộng dồn điểm
    for pt in trajectory_points:
        map_x, map_y = pt
        if 0 <= map_x < dst_width and 0 <= map_y < dst_height:
            heatmap_layer[map_y, map_x] += 1.0  

    # Làm nhòe Gaussian tạo mây nhiệt
    heatmap_blurred = cv2.GaussianBlur(heatmap_layer, (0, 0), sigmaX=25, sigmaY=25)

    # Chuẩn hóa và ép màu
    heatmap_norm = cv2.normalize(heatmap_blurred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

    # Trộn lớp nhiệt độ vào sân bóng
    final_heatmap = court_2d_clean.copy()
    mask = heatmap_norm > 2 # Hạ ngưỡng xuống để viền mây nhiệt mềm mại hơn
    
    # ĐỘ TRONG SUỐT: Giữ lại 60% màu sân gốc, phủ 40% màu nhiệt độ
    final_heatmap[mask] = cv2.addWeighted(final_heatmap, 0.6, heatmap_colored, 0.4, 0)[mask]

    # Vẽ lại vạch kẻ sân lên trên cùng lớp nhiệt
    draw_badminton_court(final_heatmap)

    # CẮT LẤY NỬA SÂN DƯỚI (Từ vị trí lưới y=335 đến cuối sân y=670)
    half_court_heatmap = final_heatmap[335:670, 0:dst_width]

    # Hiển thị và Xuất file nộp đồ án
    cv2.imshow("Heatmap Nua San (Chuan Do An)", half_court_heatmap)
    cv2.imwrite("heatmap_half_court.jpg", half_court_heatmap)
    print("Đã xuất file 'heatmap_half_court.jpg' thành công!")
    cv2.waitKey(0)

# Khởi chạy hệ thống
run_ultimate_tracker('badmintion.mp4', 'badminton_player_classifier.h5')