import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# =================================================================
# PHẦN 1: CÁC HÀM TOÁN HỌC VÀ TÌM GÓC
# =================================================================
def get_line_equation(line):
    """Tính các hệ số A, B, C cho phương trình đường thẳng Ax + By = C"""
    x1, y1, x2, y2 = line
    A = y2 - y1
    B = x1 - x2
    C = A * x1 + B * y1
    return A, B, C

def find_intersection(line1, line2):
    """Giải hệ phương trình tìm giao điểm của 2 đường thẳng"""
    A1, B1, C1 = get_line_equation(line1)
    A2, B2, C2 = get_line_equation(line2)
    
    det = A1 * B2 - A2 * B1
    if det == 0:
        return None # Hai đường thẳng song song
    
    x = (C1 * B2 - C2 * B1) / det
    y = (A1 * C2 - A2 * C1) / det
    
    # Làm tròn 4 chữ số thập phân 
    return (round(x, 4), round(y, 4))

def get_court_corners(lines):
    """Phân loại đường thẳng và tìm 4 góc sân ngoài cùng"""
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        
        if -25 < angle < 25: 
            horizontal_lines.append((x1, y1, x2, y2))
        else:
            vertical_lines.append((x1, y1, x2, y2))
            
    if not horizontal_lines or not vertical_lines:
        return None
        
    top_line = min(horizontal_lines, key=lambda l: (l[1] + l[3]) / 2)
    bottom_line = max(horizontal_lines, key=lambda l: (l[1] + l[3]) / 2)
    left_line = min(vertical_lines, key=lambda l: (l[0] + l[2]) / 2)
    right_line = max(vertical_lines, key=lambda l: (l[0] + l[2]) / 2)
    
    top_left = find_intersection(top_line, left_line)
    top_right = find_intersection(top_line, right_line)
    bottom_left = find_intersection(bottom_line, left_line)
    bottom_right = find_intersection(bottom_line, right_line)
    
    return [top_left, top_right, bottom_right, bottom_left]

# =================================================================
# PHẦN 2: HÀM XỬ LÝ ẢNH CHÍNH
# =================================================================
def detect_court_lines(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Không thể đọc được ảnh!")
        return
    height, width = img.shape[:2]
    
    # 1. Chuyển sang ảnh xám và lọc ngưỡng độ sáng
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    blurred = cv2.GaussianBlur(white_mask, (5, 5), 0)
    
    # 2. Tạo mặt nạ ROI
    mask = np.zeros_like(gray)
    polygon = np.array([[
        (int(width * 0.05), height),               
        (int(width * 0.15), int(height * 0.35)),   
        (int(width * 0.85), int(height * 0.35)),   
        (int(width * 0.95), height)                
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_blurred = cv2.bitwise_and(blurred, mask)
    
    # 3. Tìm cạnh và Biến đổi Hough
    edges = cv2.Canny(masked_blurred, threshold1=50, threshold2=150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=150, maxLineGap=50)
    
    result_img = img.copy()
    warped_court = None # Biến lưu ảnh sân đã bẻ phẳng
    matrix = None       # Biến lưu ma trận biến đổi
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        # TÌM 4 GÓC VÀ BẺ PHẲNG SÂN
        corners = get_court_corners(lines)
        if corners:
            for corner in corners:
                cv2.circle(result_img, (int(corner[0]), int(corner[1])), 15, (0, 0, 255), -1)
            
            # ===== BƯỚC MỚI: BẺ PHẲNG GÓC NHÌN (PERSPECTIVE TRANSFORM) =====
            # Chuyển đổi list 4 tọa độ thành mảng numpy float32
            # Thứ tự get_court_corners trả về đang là: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
            src_pts = np.array(corners, dtype=np.float32)
            
            # Kích thước ảnh mặt sân 2D đích (Quy đổi 1cm = 1 pixel)
            # Rộng 6.1m = 610px, Dài 13.4m = 1340px
            dst_width = 610
            dst_height = 1340
            
            dst_pts = np.array([
                [0, 0],                         # Góc trên trái
                [dst_width, 0],                 # Góc trên phải
                [dst_width, dst_height],        # Góc dưới phải
                [0, dst_height]                 # Góc dưới trái
            ], dtype=np.float32)
            
            # Tính toán ma trận biến đổi Homography
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            # Cắt và bẻ phẳng ảnh sân
            warped_court = cv2.warpPerspective(img, matrix, (dst_width, dst_height))
            print("Đã tạo thành công ma trận phối cảnh!")
            # ================================================================

    # Hiển thị
    # Mình đổi thành 4 khung hình để bạn thấy thành quả cuối cùng
    plt.figure(figsize=(20, 6))
    
    plt.subplot(1, 4, 1)
    plt.title("Lọc Vạch Trắng & ROI")
    plt.imshow(masked_blurred, cmap='gray')
    
    plt.subplot(1, 4, 2)
    plt.title("Canny Edges")
    plt.imshow(edges, cmap='gray')
    
    plt.subplot(1, 4, 3)
    plt.title("Camera Góc chéo")
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 4, 4)
    if warped_court is not None:
        plt.title("Mặt sân 2D (Top-down)")
        plt.imshow(cv2.cvtColor(warped_court, cv2.COLOR_BGR2RGB))
    else:
        plt.title("Chưa bẻ được mặt sân")
    
    plt.tight_layout()
    plt.show()

detect_court_lines('badminton_frame.jpg')