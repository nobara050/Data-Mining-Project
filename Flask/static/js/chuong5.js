let selectedColumns = []; // Biến lưu trữ các cột được chọn

// Xử lý sự kiện tải lên file CSV
function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) {
    console.log("No file selected");
    return;
  }

  // Reset các kết quả cũ
  selectedColumns = []; // Xóa danh sách các cột đã chọn
  const buttonsContainer = document.getElementById("columns-buttons-container");
  buttonsContainer.innerHTML = ""; // Xóa các nút cột cũ
  document.getElementById("table-container").innerHTML = ""; // Xóa bảng cũ
  document.getElementById("result").innerHTML = ""; // Xóa kết quả cũ
  document.querySelector(".Diagram-cluster").innerHTML = ""; // Xóa hình ảnh cũ

  const formData = new FormData();
  formData.append("file", file);

  fetch("/upload", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.table && data.columns) {
        // Hiển thị bảng từ dữ liệu trả về
        document.getElementById("table-container").innerHTML = data.table;

        // Lấy dữ liệu cột từ response và tạo các nút
        const columns = data.columns;
        createButtons(columns); // Tạo lại các nút cột
      } else if (data.error) {
        console.error("Error:", data.error);
        alert("Up File không thành công rồi"); // Hiển thị thông báo lỗi nếu có
      }
    })
    .catch((error) => {
      console.error("Error uploading file:", error);
      alert("Upload file lỗi rồi 😢");
    });
}

// Tạo các nút để chọn cột
function createButtons(columns) {
  const buttonsContainer = document.getElementById("columns-buttons-container");
  buttonsContainer.innerHTML = ""; // Reset các nút cũ

  columns.forEach((column) => {
    const button = document.createElement("button");
    button.textContent = column;
    button.classList.add("column-btn");
    button.onclick = () => toggleColumnSelection(button, column); // Tạo sự kiện cho các nút
    buttonsContainer.appendChild(button);
  });
}

// Toggle chọn cột
function toggleColumnSelection(button, column) {
  if (selectedColumns.includes(column)) {
    selectedColumns = selectedColumns.filter((col) => col !== column);
    button.classList.remove("selected");
  } else {
    selectedColumns.push(column);
    button.classList.add("selected");
  }
  console.log("Selected Columns:", selectedColumns);
}

// Xử lý K-Means khi người dùng nhấn nút
function handleKMeans() {
  const kInput = document.querySelector(".numKmean");
  const k = parseInt(kInput.value, 10);
  const fileInput = document.getElementById("csv-upload");

  if (!fileInput.files[0]) {
    alert("File upload chưa nhận, mời bạn upload file mới để gom cụm nha!");
    return;
  }

  if (isNaN(k) || k <= 0) {
    alert("Hãy nhập số K hợp lệ (lớn hơn 0)");
    return;
  }

  if (selectedColumns.length === 0) {
    alert("Bạn chưa chọn cột thuộc tính để tính gom cụm kìa!");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  formData.append("k", k);
  formData.append("columns", JSON.stringify(selectedColumns));

  fetch("http://127.0.0.1:5000/chuong5", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.clusters && data.data) {
        renderClusterInput(data.clusters, data.data);
        display3DImage(data.image); // Display the 3D plot image
        alert("Gom cụm K-Means thành công 😊!");
      } else if (data.error) {
        alert("Gom cụm xảy ra lỗi rồi 😢");
      }
    })
    .catch((error) => {
      console.error("Error:", error);
      alert("Gom cụm không thành công. Hãy thử lại lần sau nha 😢!");
    });
}

// function display3DImage(imageBase64) {
//   const diagramClusterDiv = document.querySelector(".Diagram-cluster");

//   // Create an image element and set the src to the base64-encoded image
//   const img = document.createElement("img");
//   img.src = `data:image/png;base64,${imageBase64}`;
//   img.alt = "3D Cluster Diagram";

//   // Clear previous image if any
//   diagramClusterDiv.innerHTML = "";

//   // Append the new image
//   diagramClusterDiv.appendChild(img);
// }

// function renderClusterInput(clusters, data) {
//   const resultDiv = document.getElementById("result");
//   resultDiv.innerHTML = ""; // Clear previous result

//   // Tạo một div chứa kết quả
//   const clusterContainer = document.createElement("div");

//   clusters.forEach((cluster, index) => {
//     const clusterDiv = document.createElement("div");
//     clusterDiv.classList.add("cluster");

//     // Tạo tiêu đề cho cụm
//     const clusterTitle = document.createElement("h4");
//     clusterTitle.textContent = `${cluster}:`;

//     // Tạo một input cho từng cụm, hiển thị các số thứ tự của phần tử
//     const input = document.createElement("input");
//     input.type = "text";
//     input.value = data[index].join(" "); // Hiển thị các số thứ tự của phần tử trong cụm

//     // Thêm tiêu đề và input vào div của cụm
//     clusterDiv.appendChild(clusterTitle);
//     clusterDiv.appendChild(input);

//     // Thêm div của cụm vào container
//     clusterContainer.appendChild(clusterDiv);
//   });

//   // Thêm container vào resultDiv
//   resultDiv.appendChild(clusterContainer);
// }

function renderClusterInput(clusters, data) {
  const resultDiv = document.getElementById("result");
  resultDiv.innerHTML = ""; // Clear previous result

  // Tạo một div chứa kết quả
  const clusterContainer = document.createElement("div");

  clusters.forEach((cluster, index) => {
    const clusterDiv = document.createElement("div");
    clusterDiv.classList.add("cluster");

    // Tạo tiêu đề cho cụm
    const clusterTitle = document.createElement("h4");
    clusterTitle.textContent = `${cluster}:`;

    // Tạo một input cho từng cụm, hiển thị các số thứ tự của phần tử
    const input = document.createElement("input");
    input.type = "text";
    input.value = data[index].join(" "); // Hiển thị các số thứ tự của phần tử trong cụm

    // Thêm tiêu đề và input vào div của cụm
    clusterDiv.appendChild(clusterTitle);
    clusterDiv.appendChild(input);

    // Thêm div của cụm vào container
    clusterContainer.appendChild(clusterDiv);
  });

  // Thêm container vào resultDiv
  resultDiv.appendChild(clusterContainer);
}

function display3DImage(imageData) {
  const diagramContainer = document.querySelector(".Diagram-cluster");
  diagramContainer.innerHTML = ""; // Clear any existing image

  const img = document.createElement("img");
  img.src = `data:image/png;base64,${imageData}`;
  img.alt = "KMeans Clustering 2D Plot";

  // Append the image to the Diagram-cluster div
  diagramContainer.appendChild(img);
}
