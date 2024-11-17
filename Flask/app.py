from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import chardet  # Để phát hiện encoding
from mpl_toolkits.mplot3d import Axes3D
from werkzeug.utils import secure_filename
import json
from io import BytesIO
import base64

app = Flask(__name__)

# Thư mục upload
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Hàm xóa tất cả các file trong thư mục uploads
def clear_uploads():
    if os.path.exists(UPLOAD_FOLDER):
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

# Route cho trang chính
@app.route('/')
def index():
    clear_uploads()  # Xóa file khi chuyển trang
    return render_template('index.html')

# Route cho chương 1 đến chương 4
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

@app.route('/chuong4', methods=['GET'])
def chuong4_get():
    clear_uploads()  # Xóa file khi chuyển chương
    return render_template('Chuong4.html')

@app.route('/chuong5', methods=['GET'])
def chuong5_get():
    clear_uploads()  # Xóa file khi chuyển chương
    return render_template('Chuong5.html')

# Hàm nhận dạng encoding của file
def detect_encoding(filepath):
    with open(filepath, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

# Upload file sẽ lưu lại, đọc file csv để xử lý
@app.route('/upload', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file format'}), 400

    try:
        clear_uploads()
        # Lưu file
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Phát hiện encoding file
        encoding = detect_encoding(filepath)

        # Đọc CSV với encoding đã phát hiện
        df = pd.read_csv(filepath, encoding=encoding)

        table_html = df.to_html(index=False, classes='table table-bordered', header=True)
        columns = df.columns.tolist()  # Trả về danh sách cột

        return jsonify({'table': table_html, 'columns': columns}), 200
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
@app.route('/chuong5', methods=['POST'])
def chuong5():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    k = request.form.get('k', 3)  # Default K = 3 if not provided
    columns = request.form.get('columns', '[]')  # Get selected columns as a JSON string
    selected_columns = json.loads(columns)  # Convert JSON string into Python list

    try:
        k = int(k)
    except ValueError:
        return jsonify({'error': 'K must be an integer'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file format'}), 400

    try:
        # Save the file and detect encoding
        filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(filepath)
        encoding = detect_encoding(filepath)

        # Read CSV file with detected encoding
        df = pd.read_csv(filepath, encoding=encoding)

        # Filter selected columns
        if selected_columns:
            df = df[selected_columns]

        # Apply KMeans clustering
        if k > len(df):
            return jsonify({'error': 'K cannot exceed the number of rows'}), 400

        features = df.iloc[:, :]  # Use all columns after filtering
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(features)

        # Prepare result data
        cluster_result = []
        clusters = [f"Cụm {i+1}" for i in range(k)]
        elements = df.index.tolist()  # Get the index (row numbers) of each element

        # For each cluster, collect the row numbers of elements in that cluster
        for cluster in range(k):
            cluster_row = [
                df.index[i] + 1  # Add 1 to the row index to match the 1-based numbering when exporting
                for i in range(len(df)) if df.iloc[i]['Cluster'] == cluster
            ]
            cluster_result.append(cluster_row)

        # Create 3D scatter plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot each cluster with a different color
        colors = ['r', 'g', 'b']
        for cluster in range(k):
            cluster_data = df[df["Cluster"] == cluster]
            ax.scatter(cluster_data[selected_columns[0]], cluster_data[selected_columns[1]], cluster_data[selected_columns[2]],
                       label=f"Cụm {cluster + 1}", c=colors[cluster], s=100)

        # Labels and legend
        ax.set_xlabel(selected_columns[0])
        ax.set_ylabel(selected_columns[1])
        ax.set_zlabel(selected_columns[2])
        plt.legend()
        plt.title('KMeans Clustering - 3D Plot')

        # Save the plot to a BytesIO object
        img_io = BytesIO()
        plt.savefig(img_io, format='png')
        img_io.seek(0)

        # Convert image to base64
        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

        return jsonify({
            'clusters': clusters,
            'data': cluster_result,  # Return the row indices of elements in each cluster (1-based)
            'image': img_base64  # Send the image as a base64 string
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
