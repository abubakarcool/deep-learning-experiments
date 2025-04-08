# Cinema Ticket Repurchase Prediction (ml5.js)

A simple machine learning web app using **ml5.js** to predict whether a cinema customer will **purchase a ticket again**, based on age, ticket price, seat type, genre, and whether they attended alone or in a group.

This is a **browser-based neural network** classifier built with JavaScript and trained entirely on the frontend using `ml5.js` (a wrapper for TensorFlow.js).

---

## Features

- Uses **classification neural network** with ml5.js
- Trained on cinema customer behavior data
- Predicts **repeat ticket purchase** (Yes/No)
- Clean and interactive HTML form
- Runs **entirely in the browser** — no backend required

---

## Dataset Source

### Step 1: Download Dataset

Download the dataset CSV from Kaggle:
🔗 [Cinema Hall Ticket Sales and Customer Behavior](https://www.kaggle.com/datasets/himelsarder/cinema-hall-ticket-sales-and-customer-behavior)

File name: `cinema_hall_ticket_sales.csv`

---

### Step 2: Convert CSV to JSON

1. Go to this tool: [https://csvjson.com/csv2json](https://csvjson.com/csv2json)
2. Paste the CSV content or upload the file
3. Click **“Convert”** and download the resulting JSON
4. Save the result as `data.json` and place it in your project folder

---

## How the Model Works

The model uses the following inputs:
- Age (number)
- Ticket Price (number)
- Movie Genre (Action, Comedy, Horror, Drama, Sci-Fi)
- Seat Type (Standard, Premium, VIP)
- Number of Persons (Alone or Group)

**Target**:  
`Purchase_Again` → `"Yes"` or `"No"` (binary classification)

---

## 📂 Folder Structure
```bash
cinema_purchase_prediction/ 
├── index.html 
|── Web UI 
├── script.js # ml5 neural network logic 
├── style.css # UI styling 
└── data.json # Converted dataset
```

---

## How to Run (in VS Code)

1. Open the folder in **VS Code**
2. Install the **Live Server** extension (if not already)
3. Right-click `index.html` → **Open with Live Server**
4. The app will open in your browser (`http://127.0.0.1:5500`)
5. Wait a few seconds for training to complete
6. Enter form data and click **Predict**

---

## Output
Prediction: Will Purchase Again? Yes (88.46% confidence)


---

## Tech Used

- [ml5.js](https://ml5js.org/)
- [TensorFlow.js](https://www.tensorflow.org/js)
- HTML, CSS, JavaScript
