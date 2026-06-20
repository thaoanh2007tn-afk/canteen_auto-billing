# 🍱 Canteen Auto-Billing System

Hệ thống tính tiền tự động cho căng tin trường đại học, sử dụng camera và mạng nơ-ron tích chập (CNN) để nhận diện món ăn trên khay cơm và tự động tính hóa đơn — không cần nhân viên nhập liệu thủ công.

> Bài toán đặt ra: căng tin phục vụ ~2.000 sinh viên/ngày, cần tự động hóa việc tính tiền để giảm thời gian chờ và sai sót do nhập liệu thủ công.

---

## 📂 Cấu trúc repository

| File | Vai trò |
|---|---|
| `canteen_app.py` | Mã nguồn chính — ứng dụng web Streamlit: giao diện, xử lý ảnh, gọi mô hình CNN, tính tiền và hiển thị hóa đơn |
| `train.ipynb` | Notebook xây dựng và huấn luyện mô hình CNN (chạy trên Google Colab) |
| `food_cnn.h5` | Trọng số mô hình CNN đã huấn luyện, được nạp vào ứng dụng khi khởi chạy |
| `class_names.txt` | Danh sách tên các lớp (món ăn) theo đúng thứ tự đầu ra của mô hình CNN |
| `requirements.txt` | Danh sách thư viện Python cần thiết để chạy ứng dụng |

---

## 🚀 Cài đặt & chạy thử

```bash
# 1. Clone repository
git clone https://github.com/thaoanh2007tn-afk/canteen_auto-billing.git
cd canteen_auto-billing

# 2. (Khuyến nghị) Tạo virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Cài thư viện
pip install -r requirements.txt

# 4. Chạy ứng dụng
streamlit run canteen_app.py
```

Sau khi chạy, ứng dụng sẽ mở tại `http://localhost:8501`.

> Nếu máy chưa cài TensorFlow hoặc thiếu file `food_cnn.h5`, ứng dụng tự động chuyển sang **chế độ Demo** (xem mục bên dưới) — vẫn chạy được toàn bộ giao diện và luồng xử lý.

---

## ⚙️ Cách hoạt động

```
Khay cơm (ảnh)
     │
     ▼
1) Chỉnh thông số ảnh (sáng / tương phản / màu / nét)
     │
     ▼
2) Cắt 5 vùng khay theo tỉ lệ (x, y, w, h) — có thể tinh chỉnh bằng slider
     │
     ▼
3) Mỗi vùng → CNN dự đoán tên món + độ tin cậy
     │
     ▼
4) So khớp tên món với bảng giá (PRICE_MAP)
     │
     ▼
5) Tính tổng tiền & xuất hóa đơn trên giao diện web
```

### Phân vùng khay cơm

| Vùng | x | y | w | h |
|---|---|---|---|---|
| Top-Left (món chính 1) | 3% | 3% | 44% | 48% |
| Top-Right (món chính 2) | 53% | 3% | 44% | 48% |
| Bottom-Left (món phụ 1) | 3% | 55% | 27% | 42% |
| Bottom-Center (món phụ 2) | 34% | 55% | 32% | 42% |
| Bottom-Right (món phụ 3) | 70% | 55% | 27% | 42% |

Tọa độ dùng tỉ lệ phần trăm (không phải pixel tuyệt đối) nên hoạt động đúng với mọi kích thước ảnh đầu vào (webcam hoặc ảnh tải lên), và có thể tinh chỉnh trực tiếp trên giao diện bằng thanh trượt.

### Bảng giá món ăn (16 món)

| Món ăn | Giá |
|---|---|
| Cơm trắng | 10.000đ |
| Đậu hũ sốt cà | 25.000đ |
| Cá hú kho | 30.000đ |
| Thịt kho trứng | 30.000đ |
| Thịt kho | 25.000đ |
| Canh chua có cá | 25.000đ |
| Canh chua không cá | 10.000đ |
| Sườn nướng | 30.000đ |
| Canh rau cải thảo / canh rau muống | 7.000đ |
| Rau xào (lagim / củ sắn / đậu que / đậu đũa) | 10.000đ |
| Trứng chiên | 25.000đ |
| Trứng chiên thịt | 30.000đ |

---

## 🧠 Mô hình CNN (`food_cnn.h5`)

Huấn luyện trên Google Colab, bộ dữ liệu **3.007 ảnh / 16 lớp món ăn** (2.412 train + 595 validation, tỉ lệ 80/20, seed=42). Ảnh đầu vào 128×128×3, chuẩn hóa về [0, 1].

**Kiến trúc:**

| Khối | Cấu hình |
|---|---|
| Block 1 | Conv2D(32) ×2 + BatchNorm + MaxPool + Dropout(0.25) |
| Block 2 | Conv2D(64) ×2 + BatchNorm + MaxPool + Dropout(0.25) |
| Block 3 | Conv2D(128) ×2 + BatchNorm + L2(1e-4) + MaxPool + Dropout(0.3) |
| — | GlobalAveragePooling2D |
| Dense | Dense(256, relu) + L2(1e-4) + Dropout(0.4) |
| Output | Dense(16, softmax) |

**Tổng tham số:** 325.936 (1,24 MB) — 325.040 trainable, 896 non-trainable (BatchNorm)

**Tham số huấn luyện:**
- Optimizer: Adam (lr=1e-3) + ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1e-6)
- Loss: `categorical_crossentropy` · Metric: `accuracy`
- Batch size: 16 · Max epochs: 80 (EarlyStopping patience=15, monitor=`val_accuracy`)
- Augmentation: rotation 15°, shift 10%, shear 10%, zoom 15%, horizontal flip, brightness [0.85, 1.15]
- Callbacks: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint(save_best_only=True)`

**Kết quả:** dừng ở epoch 64 — `val_accuracy` ≈ **75,6%**, `train_accuracy` ≈ **85,0%**

Chi tiết quá trình xây dựng và huấn luyện mô hình xem tại [`train.ipynb`](./train.ipynb).

---

## 🖥️ Giao diện ứng dụng

- **Nhập ảnh** — tải ảnh lên hoặc chụp trực tiếp từ webcam
- **Chỉnh thông số ảnh** — độ sáng, tương phản, màu sắc, độ sắc nét (có nút Reset)
- **Chỉnh vùng cắt** — tinh chỉnh tọa độ X/Y/W/H từng ô khay bằng thanh trượt, xem trước trực tiếp
- **Kết quả** — 5 thẻ món ăn (ảnh, tên món, % độ tin cậy, giá) + hóa đơn tổng kết

## 🛟 Chế độ Demo & xử lý lỗi

Nếu TensorFlow hoặc `food_cnn.h5` không khả dụng, ứng dụng tự chuyển sang **chế độ Demo**:
- Chọn ngẫu nhiên một món trong `class_names.txt`, độ tin cậy giả lập 65–98%
- Giao diện và toàn bộ luồng tính tiền vẫn hoạt động đầy đủ, không bị lỗi
- Flag `demo = True` hiển thị rõ trên navbar để người dùng biết kết quả là giả lập
- Mô hình được cache bằng `@st.cache_resource`, lỗi tải model được bắt và hiển thị qua `st.warning`, không làm sập ứng dụng

---

## 📌 Hạn chế hiện tại

- Chưa đếm được số lượng trứng thêm trong từng món (đã loại bỏ phần đếm trứng bằng YOLO theo yêu cầu của giảng viên) — mỗi món tính theo một mức giá cố định.
- Tọa độ 5 vùng khay giả định cố định theo loại khay hiện tại; đổi loại khay cần tinh chỉnh lại qua giao diện.
- Độ chính xác phụ thuộc vào chất lượng/số lượng ảnh huấn luyện; khuyến nghị tiếp tục bổ sung dữ liệu ở nhiều điều kiện ánh sáng và góc chụp khác nhau.

---

## 🛠️ Công nghệ sử dụng

`Streamlit` · `TensorFlow / Keras` · `OpenCV` · `Pillow` · `NumPy`

---

## 📄 License

Dự án phục vụ mục đích học tập (đồ án môn học).
