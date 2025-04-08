// Helper functions for converting inputs to numeric values
const genreToNumeric = (genre) =>
    ({ "Action": 0, "Comedy": 1, "Horror": 2, "Drama": 3, "Sci-Fi": 4 }[genre] || 0);
  
const seatToNumeric = (seat) =>
    ({ "Standard": 0, "Premium": 1, "VIP": 2 }[seat] || 0);
  
const groupToNumeric = (group) =>
    group === "Alone" ? 0 : 1;
  
const labelToBinary = (label) =>
    label === "Yes" ? 1 : 0;
  

ml5.setBackend("webgl"); // Set up the ml5 backend and neural network
  
const brain = ml5.neuralNetwork({
    task: "classification",
    debug: true,
});
  

async function loadAndProcessData() { // Load and process the data from data.json
    const response = await fetch("data.json");
    const data = await response.json();
  
    data.forEach((item) => {
        const input = {
            Age: item.Age,
            Ticket_Price: item.Ticket_Price,
            Movie_Genre: genreToNumeric(item.Movie_Genre),
            Seat_Type: seatToNumeric(item.Seat_Type),
            Group: groupToNumeric(item.Number_of_Person),
        };
        
        const output = {
            Purchase_Again: item.Purchase_Again, // Keep it as "Yes" or "No"
        };
        
        brain.addData(input, output);
    });
  
    console.log("Data loaded and processed");
}
  

function normalizeAndTrain(){ // Normalize and train the model
    brain.normalizeData();
    const trainingOptions = {
        epochs: 32,
        batchSize: 12,
    };
    // epoch	How many times we repeat all data or show the cards or flash cards to child
    // batchSize	How many items we show at one time
    brain.train(trainingOptions, finishedTraining);
}

// epochs — How Many Times the Model Sees the Full Dataset
// (Increase) Model learns more, improves accuracy => Too much = overfitting (memorizes, doesn’t generalize)	Like cramming a book too many times
// (Decrease) Faster training => Might not learn enough — underfitting	Like practicing only once before an exam

// batchSize — How Many Data Samples the Model Sees at Once Before Updating
// (Increase) Faster training => Might generalize worse, slower learning	Like studying 20 pages at once — faster, but might miss details
// (Decrease) Slower but steadier learning, better generalization => Takes longer to train	Like studying 5 pages at a time — more focus, but more time
  

function finishedTraining() { // Enable the Predict button after training
    console.log("Model training complete.");
    document.getElementById("predictBtn").disabled = false;
}
  
  
function predict() { // Predict from user input
    const input = {
        Age: parseFloat(document.getElementById("age").value),
        Ticket_Price: parseFloat(document.getElementById("price").value),
        Movie_Genre: genreToNumeric(document.getElementById("genre").value),
        Seat_Type: seatToNumeric(document.getElementById("seat").value),
        Group: groupToNumeric(document.getElementById("group").value),
    };
  
    brain.classify(input, (results) => {
        if(results && results[0]){
            const prediction = results[0].label; 
            const confidence = (results[0].confidence * 100).toFixed(2);
            document.getElementById("result").textContent =
                `Prediction: Will Purchase Again? ${prediction} (${confidence}% confidence)`;
        } else{
            document.getElementById("result").textContent = "Prediction failed.";
        }
    });
}
  

async function init() {  // Initialize everything
    try {
      await loadAndProcessData();
      normalizeAndTrain();
      document.getElementById("predictBtn").addEventListener("click", predict);
      document.getElementById("predictBtn").disabled = true;
    } catch (err) {
      console.error("Initialization error:", err);
    }
}
  
init();
  