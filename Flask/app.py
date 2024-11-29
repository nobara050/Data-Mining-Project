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
@app.route('/chuong1', methods=['GET'])
def chuong1_get():
    clear_uploads()  # Xóa file khi chuyển chương
    return render_template('Chuong1.html')

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

# Tạo mô hình Naive bayes
@app.route('/naive_bayes', methods=['POST'])
def naive_bayes():
    # Sử dụng các biến toàn cục
    global model, selected_columns, encoders  

    try:
        # Thông báo các lỗi như chưa load data lên hay chưa chọn cột thuộc tính, quyết định
        if dataset is None:
            return jsonify({'error': 'Dataset is not uploaded'}), 400

        selected_columns = request.json.get('selectedColumns', [])
        target_column = request.json.get('targetColumn', None)

        if not selected_columns or not target_column:
            return jsonify({'error': 'Selected columns or target column is missing'}), 400

        # Label Encoding
        encoders = {}
        encoded_data = dataset.copy()

        for column in selected_columns + [target_column]:
            le = LabelEncoder()
            encoded_data[column] = le.fit_transform(encoded_data[column].astype(str))
            encoders[column] = le

        features = encoded_data[selected_columns]
        target = encoded_data[target_column]

        # Train mô hình Naive Bayes
        model = GaussianNB()
        model.fit(features, target)

        # Do dữ liệu không lớn nên không slit ra train và test mà test lại trực tiếp trên dữ liệu gốc
        y_pred = model.predict(features)
        accuracy = accuracy_score(target, y_pred)
        cm = metrics.confusion_matrix(target, y_pred)

        # Tạo confusion matrix dưới dạng ảnh
        cm_df = pd.DataFrame(cm, index=encoders[target_column].classes_, columns=encoders[target_column].classes_)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, cmap="Blues", fmt="d", cbar=False)

        # Lưu ảnh vào thư mục uploads với timestamp
        uploads_dir = os.path.join(app.root_path, 'uploads')
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)

        timestamp = int(time.time())  # Lấy timestamp hiện tại
        cm_image_path = os.path.join(uploads_dir, f'confusion_matrix_{timestamp}.png')
        plt.savefig(cm_image_path)
        plt.close()
        
        # Trả về URL của ảnh đã lưu
        cm_image_url = f'/uploads/confusion_matrix_{timestamp}.png'

        return jsonify({'accuracy': accuracy, 'confusion_matrix_image_url': cm_image_url}), 200

    except Exception as e:
        return jsonify({'error': f'Error training Naive Bayes model: {str(e)}'}), 500


# Tải dữ liệu cần dự đoán bởi mô hình Naive Bayes
@app.route("/upload4_bayes", methods=["POST"])
def upload4_bayes():
    global selected_columns, model, encoders

    if not selected_columns or model is None:
        return jsonify({"error": "Model chưa được tạo. Hãy chạy bước Naive Bayes trước."}), 400

    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # Đọc file dữ liệu tải lên
        filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(filepath)

        # Phát hiện encoding
        encoding = detect_encoding(filepath)
        predict_data = pd.read_csv(filepath, encoding=encoding)

        # Encode data mới, chia ra làm 2 data để lát in ra kết quả không bị encoded
        encoded_predict_data = predict_data.copy()

        for column in selected_columns:
            if column in encoders:
                le = encoders[column]
                encoded_predict_data[column] = encoded_predict_data[column].map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
            else:
                return jsonify({"error": f"Column {column} is missing in the uploaded file."}), 400
        
        # Tiến hành dự đoán
        # predictions = model.predict(encoded_predict_data[selected_columns])
        # predict_data["Prediction"] = predictions

        # Thêm các xác suất dự đoán vào bảng predict_data
        if hasattr(model, "predict_proba"):  # Kiểm tra xem mô hình có hỗ trợ phương thức predict_proba không
            probabilities = model.predict_proba(encoded_predict_data[selected_columns])
            class_names = encoders[list(encoders.keys())[-1]].classes_  # Tên các lớp từ encoder
            for i, class_name in enumerate(class_names):
                predict_data[f"Probability_{class_name}"] = probabilities[:, i]

        # Tiến hành dự đoán
        predictions = model.predict(encoded_predict_data[selected_columns])
        predict_data["Prediction"] = predictions

        # Đưa dữ liệu dự đoán về dạng trước khi encoded
        target_column = list(encoders.keys())[-1]  # Lấy cột mục tiêu
        if target_column in encoders:
            target_encoder = encoders[target_column]
            predict_data["Prediction"] = predict_data["Prediction"].map(
                lambda x: target_encoder.inverse_transform([x])[0]
            )


        # Xuất ra HTML
        result_html = predict_data.to_html(index=False, classes="table table-bordered")

        return jsonify({"table": result_html}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Tạo mô hình Decision Tree
@app.route('/decision_tree', methods=['POST'])
def decision_tree():
    global model, selected_columns, encoders 

    try:
        selected_columns = request.json.get('selectedColumns', [])
        target_column = request.json.get('targetColumn', None)

        if not selected_columns or not target_column:
            return jsonify({'error': 'Selected columns or target column missing'}), 400

        # Tách các cột target và selected_columns ra trước
        features_data = dataset[selected_columns]
        target_data = dataset[target_column]

        # Copy features và target vào biến mới (để tránh làm thay đổi dữ liệu gốc)
        encoded_data = features_data.copy()

        # LabelEncoder lên các cột được chọn (không làm lên target)
        encoders = {}
        for column in selected_columns:
            le = LabelEncoder()
            encoded_data[column] = le.fit_transform(features_data[column].astype(str))
            encoders[column] = le

        # Tách features và target sau khi encoding
        features = encoded_data
        target = target_data

        # Train mô hình Decision Tree
        model = DecisionTreeClassifier()
        model.fit(features, target)

        # Do dữ liệu không lớn nên không slit ra train và test mà test lại trực tiếp trên dữ liệu gốc
        y_pred = model.predict(features)
        accuracy = metrics.accuracy_score(target, y_pred)

        # Tạo ảnh cây 
        dot_data = StringIO()
        export_graphviz(
            model,
            out_file=dot_data,
            filled=True,
            rounded=True,
            special_characters=True,
            feature_names=features.columns,
            class_names=[str(c) for c in target.unique()],
        )
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

        timestamp = int(time.time())
        graph_path = os.path.join(UPLOAD_FOLDER, f"decision_tree_{timestamp}.png")
        graph.write_png(graph_path)
        graph_url = f'/uploads/decision_tree_{timestamp}.png'

        return jsonify({'accuracy': accuracy, 'graph': graph_url}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Tải dữ liệu cần dự đoán bởi mô hình Decision Tree
@app.route("/upload4_decision", methods=["POST"])
def upload4_decision():
    global selected_columns, model, encoders

    if not selected_columns or model is None:
        return jsonify({"error": "Model chưa được tạo. Hãy chạy bước tạo cây trước."}), 400

    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # Đọc file CSV
        filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(filepath)

        # Tự động phát hiện encoding
        encoding = detect_encoding(filepath)
        predict_data = pd.read_csv(filepath, encoding=encoding)

        # Apply LabelEncoder to the new data
        encoded_predict_data = predict_data.copy()

        for column in selected_columns:
            if column in encoders:
                le = encoders[column]
                # Handle unseen values by mapping them to -1
                encoded_predict_data[column] = encoded_predict_data[column].map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
            else:
                return jsonify({"error": f"Column {column} is missing in the uploaded file."}), 400

        # Dự đoán kết quả
        predictions = model.predict(encoded_predict_data[selected_columns])
        predict_data["Prediction"] = predictions

        # Chuyển kết quả thành bảng HTML để hiển thị
        result_html = predict_data.to_html(index=False, classes="table table-bordered")

        return jsonify({"table": result_html}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
