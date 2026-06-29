import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# 1. ĐỌC VÀ CHUẨN BỊ DỮ LIỆU
def load_data(dataset_path):
    images = []
    labels = []
    
    # Đọc ảnh Vận động viên (Label = 1)
    player_dir = os.path.join(dataset_path, 'player')
    for img_name in os.listdir(player_dir):
        img_path = os.path.join(player_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (64, 128)) # Đảm bảo đúng size
            images.append(img)
            labels.append(1)
            
    # Đọc ảnh Rác/Nhiễu (Label = 0)
    not_player_dir = os.path.join(dataset_path, 'not_player')
    for img_name in os.listdir(not_player_dir):
        img_path = os.path.join(not_player_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (64, 128))
            images.append(img)
            labels.append(0)
            
    # Chuyển đổi sang mảng Numpy và Chuẩn hóa pixel về khoảng [0, 1]
    X = np.array(images, dtype='float32') / 255.0
    y = np.array(labels)
    
    return X, y

print("Đang nạp dữ liệu...")
X, y = load_data('dataset')
print(f"Tổng số ảnh: {len(X)}")

# Chia dữ liệu: 80% để train, 20% để test (kiểm tra chéo)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. XÂY DỰNG KIẾN TRÚC MẠNG CNN
model = Sequential([
    # Lớp tích chập 1: Tìm kiếm các góc cạnh, đường nét cơ bản
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Lớp tích chập 2: Tìm kiếm các khối hình phức tạp hơn (tay, chân, thân người)
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Làm phẳng mảng 2D thành 1D
    Flatten(),
    
    # Mạng nơ-ron truyền thống để ra quyết định
    Dense(64, activation='relu'),
    Dropout(0.5), # Tắt ngẫu nhiên 50% nơ-ron để chống học vẹt (Overfitting)
    Dense(1, activation='sigmoid') # Đầu ra 1 nơ-ron duy nhất (0 hoặc 1)
])

# Chỉ định thuật toán tối ưu và hàm tính lỗi
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. TIẾN HÀNH HUẤN LUYỆN
print("Bắt đầu huấn luyện mô hình...")
# Class weights giúp mô hình chú ý hơn vào tập 'not_player' vì nó có ít ảnh hơn
class_weights = {0: 6.0, 1: 1.0} 

history = model.fit(
    X_train, y_train,
    epochs=10, # Chạy lặp lại 10 vòng
    batch_size=16,
    validation_data=(X_test, y_test),
    class_weight=class_weights
)

# 4. LƯU MÔ HÌNH
# Đây chính là file bạn sẽ nộp lên Github!
model.save('badminton_player_classifier.h5')
print("Đã lưu mô hình thành công vào file 'badminton_player_classifier.h5'!")