# Bài 1: Hệ thống đề xuất sách (Doanh)
- Xây dựng hệ thống đề xuất sách dự trên đánh giá của người dùng
# Bài 2: Phân cụm người dùng (Khoa)
- Phân nhóm người dùng dựa trên hành vi đọc sách và đánh giá

Book Recommendation System (Ratings-based)
1. Giới thiệu

Dự án xây dựng hệ thống gợi ý sách dựa trên dữ liệu đánh giá người dùng (Collaborative Filtering).
Mục tiêu: với một User-ID, hệ thống sẽ đề xuất Top-N sách mà người dùng chưa đánh giá nhưng có khả năng thích cao.

Pipeline theo yêu cầu môn học:
Preprocessing → EDA → Model → Evaluation → Demo

2. Dataset

Nguồn dữ liệu: Kaggle – Books Dataset
Link: https://www.kaggle.com/datasets/saurabhbagchi/books-dataset/data

Sử dụng 2 file chính:

Books.csv (metadata sách): ISBN, Book-Title, Book-Author, Year-Of-Publication, Publisher, Image URLs...

Ratings.csv (đánh giá): User-ID, ISBN, Book-Rating

Lưu ý: Không commit trực tiếp dataset lớn lên GitHub. Chỉ cần để link Kaggle và hướng dẫn tải/chạy.

3. Cấu trúc thư mục
<project_name>/
├─ README.md
├─ Nx_report.pdf
├─ requirements.txt
├─ data/
│  ├─ raw/            # (tuỳ chọn) chứa dữ liệu thô nếu bạn tải về local
│  └─ processed/      # (tuỳ chọn) chứa dữ liệu sau xử lý / output
├─ notebook/
│  └─ books-recommend_f.ipynb  # notebook chạy demo (EDA + Train + Evaluate)
├─ src/
│  ├─ preprocessing.py         # xử lý dữ liệu
│  ├─ eda.py                   # phân tích EDA + vẽ biểu đồ
│  ├─ feature_engineering.py   # (tuỳ chọn) tạo đặc trưng nâng cao
│  ├─ model_<hovaten>.py       # huấn luyện mô hình (MF / baseline)
│  ├─ evaluation.py            # đánh giá mô hình (RMSE/MAE, Precision/Recall@K)
│  └─ main.py                  # chạy pipeline end-to-end
└─ web/                        # (tuỳ chọn) nếu làm giao diện

4. Ý tưởng & phương pháp
4.1 Baseline (Popularity-based)

Xếp hạng sách theo độ phổ biến dựa trên:

số lượt rating (rating_count)

điểm trung bình (avg_rating)

Baseline giúp có “mốc tham chiếu” và hoạt động tốt trong trường hợp cold-start.

4.2 Mô hình chính (Collaborative Filtering – Matrix Factorization)

Sử dụng Matrix Factorization (PyTorch):

Mã hoá user và item thành embedding

Dự đoán rating bằng tích vô hướng + bias

Huấn luyện với loss MSE, tối ưu Adam

4.3 Đánh giá

RMSE, MAE: đánh giá độ chính xác dự đoán rating

Precision@K, Recall@K: đánh giá chất lượng gợi ý Top-K (với threshold, ví dụ rating ≥ 7)

5. Hướng dẫn chạy (Local)
5.1 Cài thư viện
pip install -r requirements.txt

5.2 Chạy pipeline end-to-end
python src/main.py


Nếu dataset ở local, bạn có thể sửa đường dẫn trong src/main.py hoặc set biến môi trường.

Ví dụ set biến môi trường:

# Windows (PowerShell)
$env:BOOKS_PATH="data/raw/Books.csv"
$env:RATINGS_PATH="data/raw/Ratings.csv"
python src/main.py

# Linux/Mac
export BOOKS_PATH="data/raw/Books.csv"
export RATINGS_PATH="data/raw/Ratings.csv"
python src/main.py

6. Hướng dẫn chạy trên Kaggle
Cách 1: Chạy notebook

Mở file trong notebook/ và Run All.

Cách 2: Chạy bằng script

Trong Kaggle Notebook:

!pip -q install -r requirements.txt
!python src/main.py


Đường dẫn dataset Kaggle (mặc định trong src/main.py):

/kaggle/input/books-dataset/books_data/books.csv

/kaggle/input/books-dataset/books_data/ratings.csv

7. Output (kết quả sinh ra)

Sau khi chạy, hệ thống tạo các file (tuỳ theo bạn bật/tắt phần lưu):

eda_analysis.png
Biểu đồ EDA: phân phối rating, số rating theo user/sách, phân phối rating trung bình

training_history.png
Lịch sử huấn luyện (loss/RMSE/MAE theo epoch)

baseline_popular_books.csv
Top sách theo baseline popularity

evaluation_metrics.csv
Tổng hợp RMSE/MAE và Precision@K/Recall@K

sample_recommendations.csv
Demo Top-N sách đề xuất cho một User-ID

matrix_factorization_model.pth (tuỳ chọn)
Trọng số mô hình sau huấn luyện

Khuyến nghị: Nên commit các file png/csv (nhẹ) để người chấm thấy kết quả; không cần commit file model .pth nếu nặng.

8. Mô tả nhanh các file code

src/preprocessing.py: đọc dữ liệu, clean, lọc rating>0, lọc sparsity theo ngưỡng, encode user/item, chia train/test theo user

src/eda.py: thống kê và vẽ biểu đồ EDA, lưu eda_analysis.png

src/model_<hovaten>.py: baseline popularity + MF (PyTorch) + hàm recommend Top-N

src/evaluation.py: tính RMSE/MAE và Precision/Recall@K

src/main.py: chạy toàn bộ pipeline, lưu kết quả ra csv/png


