let selectedColumns = []; // Biến lưu trữ các cột được chọn

// Xử lý sự kiện tải lên file CSV
function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) {
    console.log("No file selected");
    return;
  }

  // Reset selectedColumns mỗi khi tải lên file mới
  selectedColumns = [];
  const buttonsContainer = document.getElementById("columns-buttons-container");
  buttonsContainer.innerHTML = ""; // Reset các nút cũ

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
        alert(data.error); // Hiển thị thông báo lỗi nếu có
      }
    })
    .catch((error) => {
      console.error("Error uploading file:", error);
      alert("File upload failed. Please try again.");
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
