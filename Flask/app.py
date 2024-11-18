from flask import Flask, request, jsonify, render_template, send_from_directory
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from werkzeug.utils import secure_filename
import chardet  # Để phát hiện encoding
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from IPython.display import Image
import matplotlib.pyplot as plt
from sklearn import metrics
from io import StringIO
from io import BytesIO
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import pydotplus
import graphviz
import base64
import atexit
import time
import json
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

app = Flask(__name__)

def cleanup():
    import matplotlib.pyplot as plt
    plt.close('all')  # Close all figures to avoid issues with Matplotlib in Flask
atexit.register(cleanup)

# Biến lưu trữ cây quyết định toàn cục
clf = None
updated_selected_columns = []
# encoders = {}
# dataset = None
# Thư mục upload
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Hàm xóa tất cả các file trong thư mục uploads (bao gồm ảnh)
def clear_uploads():
    if os.path.exists(UPLOAD_FOLDER):
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

# Favicon
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        app.static_folder, 'favicon.ico', mimetype='image/vnd.microsoft.icon'
    )

@app.route('/uploads/<filename>')
def serve_upload(filename):
    upload_folder = os.path.join(app.root_path, 'uploads')
    return send_from_directory(upload_folder, filename)

# Route cho trang chính
@app.route('/')
def index():
    clear_uploads()  # Xóa file khi chuyển trang
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
    global dataset, clf, updated_selected_columns  # Make clf and updated_selected_columns global

    # Reset model and selected columns on each upload
    clf = None
    updated_selected_columns = []

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

# Route to create decision tree model
@app.route('/decision_tree', methods=['POST'])
def decision_tree():
    global clf, updated_selected_columns, encoders  # Add encoders as global

    try:
        selected_columns = request.json.get('selectedColumns', [])
        target_column = request.json.get('targetColumn', None)

        if not selected_columns or not target_column:
            return jsonify({'error': 'Selected columns or target column missing'}), 400

        # Prepare dataset
        encoders = {}
        encoded_data = dataset.copy()

        # Apply LabelEncoder to selected columns only (not target column)
        for column in selected_columns:
            le = LabelEncoder()
            encoded_data[column] = le.fit_transform(encoded_data[column].astype(str))
            encoders[column] = le

        # Separate features and target column (without encoding target)
        features = encoded_data[selected_columns]
        target = encoded_data[target_column]  # Don't apply LabelEncoder to target column here

        # Train Decision Tree
        clf = DecisionTreeClassifier()
        clf.fit(features, target)

        # Update global selected columns
        updated_selected_columns = selected_columns

        # Calculate accuracy
        y_pred = clf.predict(features)
        accuracy = metrics.accuracy_score(target, y_pred)

        # Visualize Decision Tree
        dot_data = StringIO()
        export_graphviz(
            clf,
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


# Route to upload new data and make predictions
@app.route("/upload4", methods=["POST"])
def upload4():
    global updated_selected_columns, clf, encoders

    if not updated_selected_columns or clf is None:
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
        new_data = pd.read_csv(filepath, encoding=encoding)

        # Apply LabelEncoder to the new data
        encoded_new_data = new_data.copy()

        for column in updated_selected_columns:
            if column in encoders:
                le = encoders[column]
                # Handle unseen values by mapping them to -1
                encoded_new_data[column] = encoded_new_data[column].map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
            else:
                return jsonify({"error": f"Column {column} is missing in the uploaded file."}), 400

        # Dự đoán kết quả
        predictions = clf.predict(encoded_new_data[updated_selected_columns])
        new_data["Prediction"] = predictions

        # Chuyển kết quả thành bảng HTML để hiển thị
        result_html = new_data.to_html(index=False, classes="table table-bordered")

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

        # Create 2D scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot each cluster with a different color
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # You can add more colors as needed
        for cluster in range(k):
            cluster_data = df[df["Cluster"] == cluster]
            ax.scatter(cluster_data[selected_columns[0]], cluster_data[selected_columns[1]],
                       label=f"Cụm {cluster + 1}", c=colors[cluster % len(colors)], s=100)

        # Labels and legend
        ax.set_xlabel(selected_columns[0])
        ax.set_ylabel(selected_columns[1])
        plt.legend()
        plt.title('KMeans Clustering - 2D Plot')

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
