// Xử lý K-Means khi người dùng nhấn nút
function handleKMeans() {
  const kInput = document.querySelector(".numKmean");
  const k = parseInt(kInput.value, 10);
  const fileInput = document.getElementById("csv-upload");

  if (!fileInput.files[0]) {
    alert("Please upload a file before running K-Means");
    return;
  }

  if (isNaN(k) || k <= 0) {
    alert("Please enter a valid K value (must be greater than 0)");
    return;
  }

  if (selectedColumns.length === 0) {
    alert("Please select columns for K-Means clustering");
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
        alert("K-Means clustering completed successfully!");
      } else if (data.error) {
        alert(data.error);
      }
    })
    .catch((error) => {
      console.error("Error:", error);
      alert("Failed to perform K-Means. Please try again.");
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
