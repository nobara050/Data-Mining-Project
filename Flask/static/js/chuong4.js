window.runDecisionTree = function () {
  const targetColumn = document.getElementById("target-column").value;

  if (!targetColumn) {
    alert("Target column is required.");
    return;
  }

  if (selectedColumns.length === 0) {
    alert("Please select at least one column.");
    return;
  }

  fetch("/decision_tree", {
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
        alert(data.error);
      } else {
        alert("Model Accuracy: " + data.accuracy);
        document.getElementById("decision-tree-result").innerHTML = `
                  <h3>Decision Tree Visualization:</h3>
                  <img src="${data.graph}" alt="Decision Tree" />
              `;
      }
    })
    .catch((error) => {
      console.error("Error:", error);
      alert("An error occurred while running the decision tree.");
    });
};
