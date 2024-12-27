document.getElementById('predictionForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const modelSelect = document.getElementById('modelSelect').value;
    const timeframeSelect = document.getElementById('timeframeSelect').value;
    const paramsInput = document.getElementById('paramsInput').value;
    const dataInput = document.getElementById('dataInput').value;
    const dataArray = dataInput.split(',').map(Number);

    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            model: modelSelect,
            timeframe: timeframeSelect,
            parameters: paramsInput.split(',').map(Number),
            data: dataArray
        }),
    });

    const result = await response.json();
    document.getElementById('result').textContent = `Prediction: ${result.prediction}`;
});
