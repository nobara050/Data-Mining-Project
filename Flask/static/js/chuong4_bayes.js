let selectedColumns = []; // Lưu trữ các cột được chọn

function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) {
    console.log("No file selected");
    return;
  }

  resetResults();
  selectedColumns = [];
  const buttonsContainer = document.getElementById("columns-buttons-container");
  buttonsContainer.innerHTML = ""; // Reset the buttons

  const formData = new FormData();
  formData.append("file", file);

  fetch("/upload", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.table && data.columns) {
        // Display the table
        document.getElementById("table-container").innerHTML = data.table;

        // Get columns and create the buttons
        const columns = data.columns;
        createButtons(columns); // Create the column buttons

        // Populate the ComboBox with columns for target selection
        populateComboBox(columns); // Populate ComboBox with columns
      } else if (data.error) {
        console.error("Error:", data.error);
        alert("Lỗi không load được combobox 😢");
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

function populateComboBox(columns) {
  const targetColumnSelect = document.getElementById("target-column");
  targetColumnSelect.innerHTML = ""; // Clear the previous options

  // Add an option for each column
  columns.forEach((column) => {
    const option = document.createElement("option");
    option.value = column;
    option.textContent = column;
    targetColumnSelect.appendChild(option);
  });
}

function createTargetColumnDropdown(columns) {
  const targetColumnSelect = document.getElementById("target-column");

  // Clear existing options
  targetColumnSelect.innerHTML = "";

  // Add a default placeholder option
  const placeholderOption = document.createElement("option");
  placeholderOption.textContent = "Select target column";
  targetColumnSelect.appendChild(placeholderOption);

  // Add options for all columns
  columns.forEach((column) => {
    const option = document.createElement("option");
    option.value = column;
    option.textContent = column;
    targetColumnSelect.appendChild(option);
  });
}

function runNaiveBayes() {
  const targetColumn = document.getElementById("target-column").value;

  if (!targetColumn) {
    alert("Vui lòng chọn thuộc tính quyết định!");
    return;
  }

  if (selectedColumns.length === 0) {
    alert("Vui lòng chọn ít nhất một thuộc tính phân lớp!");
    return;
  }

  if (selectedColumns.includes(targetColumn)) {
    alert("Thuộc tính quyết định không được trùng với thuộc tính phân lớp!");
    return;
  }

  fetch("/naive_bayes", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      selectedColumns: selectedColumns,
      targetColumn: targetColumn,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.error) {
        alert(`Lỗi xử lý: ${data.error}`);
        return;
      }

      // Hiển thị kết quả
      document.getElementById(
        "accuracy-input"
      ).value = `Độ chính xác: ${data.accuracy}`;

      // Hiển thị Confusion Matrix dưới dạng ảnh
      document.getElementById("confusion-matrix-container").innerHTML = `
            <h3>Confusion Matrix:</h3>
            <img src="${data.confusion_matrix_image_url}" alt="Confusion Matrix Image" />
          `;
    })
    .catch((error) => {
      console.error("Lỗi khi xử lý Naive Bayes:", error);
      alert("Có lỗi xảy ra khi chạy Naive Bayes!");
    });
}

function resetResults() {
  document.getElementById("table-container").innerHTML = "";
  document.getElementById("columns-buttons-container").innerHTML = "";
  document.getElementById("target-column").innerHTML = "";
  document.getElementById("accuracy-input").value = "";
  document.getElementById("confusion-matrix-container").innerHTML = "";
  document.getElementById("new-prediction-table-container").innerHTML = "";
}

function handleFileUploadNew(event) {
  const file = event.target.files[0];
  if (!file) {
    alert("Chưa chọn file để dự đoán");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  fetch("/upload4_bayes", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.error) {
        alert(data.error);
        // File có vấn đề hoặc có lỗi xảy ra rồi 😢
      } else {
        document.getElementById("new-prediction-table-container").innerHTML = `
            <h3>Kết quả dự đoán:</h3>
            ${data.table || "Không có dữ liệu"}
          `;
      }
    })
    .catch((error) => {
      alert("Có lỗi xảy ra khi tải file dự đoán.");
    });
}
