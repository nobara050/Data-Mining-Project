from flask import Flask, request, jsonify, render_template, send_from_directory
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from werkzeug.utils import secure_filename
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from IPython.display import Image
import matplotlib.pyplot as plt
from sklearn import metrics
from io import StringIO
from io import BytesIO
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pydotplus
import graphviz
import chardet
from sklearn.preprocessing import OneHotEncoder

import atexit   
import base64
import time
import json
import os

os.environ['OMP_NUM_THREADS'] = '1'
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

app = Flask(__name__)


# ================================================================================
# ==================                                        ======================
# ==================              CÁC HÀM PHỤ TRỢ           ======================
# ==================                                        ======================
# ================================================================================

# Tránh lỗi Matplotlib
def cleanup():
    import matplotlib.pyplot as plt
    plt.close('all')  
atexit.register(cleanup)

# Biến lưu trữ cây quyết định toàn cục dùng trong chương 4
selected_columns = []
dataset = None
encoders = {}
model = None

# Định nghĩa đường dẫn thư mục upload
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Hàm xóa tất cả các file trong thư mục uploads (bao gồm ảnh)
def clear_uploads():
    if os.path.exists(UPLOAD_FOLDER):
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

# Load Favicon lên web
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        app.static_folder, 'favicon.ico', mimetype='image/vnd.microsoft.icon'
    )


# Upload file lên
@app.route('/uploads/<filename>')
def serve_upload(filename):
    upload_folder = os.path.join(app.root_path, 'uploads')
    return send_from_directory(upload_folder, filename)



# ================================================================================
# ==================                                        ======================
# ==================        ROUTE CÁC TRANG METHOD GET      ======================
# ==================                                        ======================
# ================================================================================



# Route cho trang chính
@app.route('/')
def index():
    clear_uploads()  # Xóa file khi chuyển chương
    return render_template('index.html')

# Route cho chương 1 đến chương 5
@app.route('/chuong1_pearson', methods=['GET'])
def chuong1_peaerson_get():
    clear_uploads()  # Xóa file khi chuyển chương
    return render_template('Chuong1_pearson.html')

@app.route('/chuong1_binning', methods=['GET'])
def chuong1_binning_get():
    clear_uploads()  # Xóa file khi chuyển chương
    return render_template('Chuong1_binning.html')

@app.route('/chuong2', methods=['GET'])
def chuong2_get():
    clear_uploads()  # Xóa file khi chuyển chương
    return render_template('Chuong2.html')

@app.route('/chuong3', methods=['GET'])
def chuong3_get():
    clear_uploads()  # Xóa file khi chuyển chương
    return render_template('Chuong3.html')

@app.route('/chuong4_gini', methods=['GET'])
def chuong4_gini_get():
    clear_uploads()  # Xóa file khi chuyển chương
    return render_template('Chuong4_gini.html')

@app.route('/chuong4_bayes', methods=['GET'])
def chuong4_bayes_get():
    clear_uploads()  # Xóa file khi chuyển chương
    return render_template('Chuong4_bayes.html')

@app.route('/chuong5', methods=['GET'])
def chuong5_get():
    clear_uploads()  # Xóa file khi chuyển chương
    return render_template('Chuong5.html')

# Hàm nhận dạng encoding của file
def detect_encoding(filepath):
    with open(filepath, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']


# ================================================================================
# ==================                                        ======================
# ==================     HÀM LOAD FILE CSV LÊN ĐỂ SỬ DỤNG   ======================
# ==================                                        ======================
# ================================================================================

# Upload file sẽ lưu lại, đọc file csv để xử lý và xóa toàn bộ file trong uploads 
# (reset bộ nhớ chứ không mỗi lần người dùng chọn một dataset mới lưu lại sẽ tràn)
@app.route('/upload', methods=['POST'])
def upload_csv():
    # Reset các biến toàn cục mỗi lần load file
    global dataset, model, selected_columns, encoders, dataset 
    selected_columns = []
    dataset = None
    encoders = {}
    model = None
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file format'}), 400

    try:
        clear_uploads()
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        encoding = detect_encoding(filepath)
        df = pd.read_csv(filepath, encoding=encoding)

        table_html = df.to_html(index=False, classes='table table-bordered', header=True)
        columns = df.columns.tolist()

        # Save DataFrame globally
        dataset = df

        return jsonify({'table': table_html, 'columns': columns}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==========================================
# ==========================================
# =====                               ======
# =====          CHƯƠNG 4             ======
# =====                               ======
# ==========================================
# ==========================================

@app.route('/upload_gini', methods=['POST'])
def upload_csv_gini():
    global dataset_gini, model_gini
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file format'}), 400

    try:
        clear_uploads()
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        encoding = detect_encoding(filepath)
        df = pd.read_csv(filepath, encoding=encoding)

        table_html = df.to_html(index=False, classes='table table-bordered', header=True)
        columns = df.columns.tolist()
        data = df.values.tolist()

        dataset_gini = df

        return jsonify({'table': table_html, 'columns': columns, 'data': data}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/predict_gini', methods=['POST'])
def predict_gini():
    global dataset_gini, model_gini

    try:
        data = request.get_json()
        selected_values = data.get("selected_values")

        combobox_df = pd.DataFrame([selected_values], columns=dataset_gini.columns[:-1])

        dataset_no_target = dataset_gini.drop(columns=[dataset_gini.columns[-1]])

        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoded_data = encoder.fit_transform(dataset_no_target)
        pre_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(dataset_no_target.columns))

        target = dataset_gini[dataset_gini.columns[-1]]
        label_encoder = LabelEncoder()
        target_encoded = label_encoder.fit_transform(target)

        X = pre_df
        y = target_encoded

        model_gini = DecisionTreeClassifier(criterion='gini')
        model_gini.fit(X, y)

        combobox_df_pre = encoder.transform(combobox_df)
        combobox_df_pre = pd.DataFrame(combobox_df_pre, columns=encoder.get_feature_names_out(combobox_df.columns))

        missing_cols = set(X.columns) - set(combobox_df_pre.columns)
        for col in missing_cols:
            combobox_df_pre[col] = 0

        prediction_proba = model_gini.predict_proba(combobox_df_pre)
        prediction = prediction_proba[0]

        target_columns = label_encoder.classes_.tolist()

        return jsonify({
            'prediction': {
                target_columns[0]: f"{(prediction[0] * 100):.2f}%",
                target_columns[1]: f"{(prediction[1] * 100):.2f}%"
            }
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/upload_naive_bayes', methods=['POST'])
def upload_naive_bayes():
    global dataset, model, encoders
    encoders = {}
    model = None
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file format'}), 400

    try:
        clear_uploads()
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        encoding = detect_encoding(filepath)
        df = pd.read_csv(filepath, encoding=encoding)

        table_html = df.to_html(index=False, classes='table table-bordered', header=True)
        columns = df.columns.tolist()  # Lưu cột trong DataFrame
        data = df.values.tolist()  # Dữ liệu thô

        dataset = df

        return jsonify({'table': table_html, 'columns': columns, 'data': data}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_naive_bayes', methods=['POST'])
def predict_naive_bayes():
    global dataset, model

    try:
        data = request.get_json()
        selected_values = data.get("selected_values")

        # Tạo DataFrame từ selected_values (các giá trị combobox)
        combobox_df = pd.DataFrame([selected_values], columns=dataset.columns[:-1])  # Loại bỏ cột target

        # Tiền xử lý dữ liệu
        dataset_no_target = dataset.drop(columns=[dataset.columns[-1]])  # Loại bỏ cột mục tiêu trước khi encode

        # Áp dụng OneHotEncoder cho các cột tính năng (X)
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoded_data = encoder.fit_transform(dataset_no_target)
        pre_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(dataset_no_target.columns))

        # Sử dụng LabelEncoder cho cột mục tiêu (y)
        target = dataset[dataset.columns[-1]]
        label_encoder = LabelEncoder()
        target_encoded = label_encoder.fit_transform(target)  # Mã hóa cột mục tiêu thành các số nguyên

        # Chia X (tính năng) và y (cột mục tiêu đã được mã hóa)
        X = pre_df
        y = target_encoded  # y là mảng 1D với giá trị số nguyên

        # Huấn luyện mô hình với Laplace Smoothing (var_smoothing)
        model = GaussianNB(var_smoothing=1e-9)  # Chỉnh var_smoothing nếu cần làm trơn (Laplace Smoothing)
        model.fit(X, y)

        # Tiền xử lý dữ liệu combobox (giống như đã làm với train data)
        combobox_df_pre = encoder.transform(combobox_df)
        combobox_df_pre = pd.DataFrame(combobox_df_pre, columns=encoder.get_feature_names_out(combobox_df.columns))

        # Kiểm tra nếu có cột thiếu và thêm chúng với giá trị 0
        missing_cols = set(X.columns) - set(combobox_df_pre.columns)
        for col in missing_cols:
            combobox_df_pre[col] = 0  # Thêm cột thiếu với giá trị 0 (nếu có)

        # Dự đoán với mô hình đã huấn luyện
        prediction = model.predict_proba(combobox_df_pre)

        # Lấy xác suất cho từng lớp
        prediction = prediction[0]  # Lấy kết quả cho một mẫu duy nhất

        # Lấy tên các lớp (Cao, Thấp) từ label_encoder
        target_columns = label_encoder.classes_.tolist()

        # Trả về kết quả dự đoán và xác suất của từng lớp, chuyển đổi thành phần trăm
        return jsonify({
            'prediction': {
                target_columns[0]: f"{prediction[0] * 100:.2f}%",  # Lớp Cao
                target_columns[1]: f"{prediction[1] * 100:.2f}%"   # Lớp Thấp
            }
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500








# ==========================================
# ==========================================
# =====                               ======
# =====          CHƯƠNG 5             ======
# =====                               ======
# ==========================================
# ==========================================

# Khi bấm nút Gom cụm trong chương 5
from sklearn.decomposition import PCA

@app.route('/chuong5', methods=['POST'])
def chuong5():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    k = request.form.get('k', 3)  # Mặc định K = 3 nếu không được cung cấp
    columns = request.form.get('columns', '[]')  # Lấy các cột được chọn dưới dạng chuỗi JSON
    selected_columns = json.loads(columns)  # Chuyển chuỗi JSON thành danh sách Python

    try:
        k = int(k)
    except ValueError:
        return jsonify({'error': 'K must be an integer'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file format'}), 400

    try:
        # Lưu file và phát hiện mã hóa
        filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(filepath)
        encoding = detect_encoding(filepath)

        # Đọc file CSV với mã hóa đã phát hiện
        df = pd.read_csv(filepath, encoding=encoding)

        # Lọc các cột đã chọn
        if selected_columns:
            df = df[selected_columns]

        # Áp dụng phân cụm KMeans
        if k > len(df):
            return jsonify({'error': 'K cannot exceed the number of rows'}), 400

        features = df.iloc[:, :]  # Sử dụng tất cả các cột sau khi lọc
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(features)

        # Giảm chiều dữ liệu xuống 2D bằng PCA
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(features)  # Giảm chiều dữ liệu
        df['PCA1'] = pca_components[:, 0]  # Thành phần chính 1
        df['PCA2'] = pca_components[:, 1]  # Thành phần chính 2

        # Chuẩn bị dữ liệu kết quả
        cluster_result = []
        clusters = [f"Cụm {i+1}" for i in range(k)]  # Tạo tên các cụm
        elements = df.index.tolist()  # Lấy danh sách chỉ mục (số dòng) của từng phần tử

        # Đối với mỗi cụm, thu thập các chỉ mục dòng của các phần tử trong cụm đó
        for cluster in range(k):
            cluster_row = [
                df.index[i] + 1  # Thêm 1 vào chỉ mục dòng để tương ứng với số dòng bắt đầu từ 1 khi xuất dữ liệu
                for i in range(len(df)) if df.iloc[i]['Cluster'] == cluster
            ]
            cluster_result.append(cluster_row)

        # Tạo biểu đồ phân tán 2D
        fig, ax = plt.subplots(figsize=(10, 8))

        # Vẽ từng cụm với màu sắc khác nhau
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Bạn có thể thêm nhiều màu hơn nếu cần
        for cluster in range(k):
            cluster_data = df[df["Cluster"] == cluster]
            ax.scatter(cluster_data['PCA1'], cluster_data['PCA2'],
                       label=f"Cụm {cluster + 1}", c=colors[cluster % len(colors)], s=100)

        # Thêm nhãn và chú thích
        ax.set_xlabel('PCA1')
        ax.set_ylabel('PCA2')
        plt.legend()
        plt.title('KMeans Clustering - 2D Plot (PCA)')

        # Lưu biểu đồ vào đối tượng BytesIO
        img_io = BytesIO()
        plt.savefig(img_io, format='png')
        img_io.seek(0)

        # Chuyển đổi ảnh sang định dạng base64
        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

        return jsonify({
            'clusters': clusters,  # Trả về tên các cụm
            'data': cluster_result,  # Trả về các chỉ mục dòng của các phần tử trong mỗi cụm (dựa trên chỉ mục bắt đầu từ 1)
            'image': img_base64  # Gửi ảnh dưới dạng chuỗi base64
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint xử lý khi nhấn nút
@app.route("/exit", methods=["POST"])
def exit_program():
    print("Exiting program gracefully...")
    os._exit(0)  # Dừng Flask server và toàn bộ chương trình


if __name__ == '__main__':
    app.run(debug=True)
