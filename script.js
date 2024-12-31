let model, tokenizer, categoryNames;

async function loadModelAndTokenizer() {
  console.log("Loading model...");

  try {
    const response = await fetch("./model.json");
    if (!response.ok) {
      throw new Error(
        "model.json not found or inaccessible. Check the file path.",
      );
    }

    console.log("model.json exists and is accessible.");

    const modelJson = await response.json();
    console.log("Content of model.json:", modelJson);

    model = await tf.loadLayersModel("./model.json");
    console.log("Model loaded successfully:", model);
    document.getElementById("modelStatus").innerText = "Loaded Successfully";
  } catch (error) {
    console.error("Error loading model:", error.message);
    document.getElementById("modelStatus").innerText = "Failed to Load";
    const modelErrorEl = document.getElementById("modelError");
    modelErrorEl.style.display = "block";
    modelErrorEl.innerText = `Model Load Error: ${error.message}`;
  }

  console.log("Loading tokenizer...");
  try {
    const response = await fetch("./tokenizer.json");
    if (!response.ok) {
      throw new Error(
        "tokenizer.json not found or inaccessible. Check the file path.",
      );
    }

    console.log("tokenizer.json exists and is accessible.");

    const tokenizerConfig = await response.json();
    console.log("Content of tokenizer.json:", tokenizerConfig);

    tokenizer = new Tokenizer(tokenizerConfig);
    console.log("Tokenizer loaded successfully.");
    document.getElementById("tokenizerStatus").innerText =
      "Loaded Successfully";

    categoryNames = [
      "alt.atheism",
      "comp.graphics",
      "comp.os.ms-windows.misc",
      "comp.sys.ibm.pc.hardware",
      "comp.sys.mac.hardware",
      "comp.windows.x",
      "misc.forsale",
      "rec.autos",
      "rec.motorcycles",
      "rec.sport.baseball",
      "rec.sport.hockey",
      "sci.crypt",
      "sci.electronics",
      "sci.med",
      "sci.space",
      "soc.religion.christian",
      "talk.politics.guns",
      "talk.politics.mideast",
      "talk.politics.misc",
      "talk.religion.misc",
    ];
    console.log("Category names loaded:", categoryNames);
    displayCategories(categoryNames);
  } catch (error) {
    console.error("Error loading tokenizer:", error.message);
    document.getElementById("tokenizerStatus").innerText = "Failed to Load";
    const tokenizerErrorEl = document.getElementById("tokenizerError");
    tokenizerErrorEl.style.display = "block";
    tokenizerErrorEl.innerText = `Tokenizer Load Error: ${error.message}`;
  }
}

function displayCategories(categories) {
  const categoryListEl = document.getElementById("categoryList");
  categoryListEl.innerHTML = "";
  categories.forEach((category) => {
    const listItem = document.createElement("li");
    listItem.innerText = category;
    categoryListEl.appendChild(listItem);
  });
}

class Tokenizer {
  constructor(config) {
    this.wordIndex = config.word_index || {};
  }

  textsToSequences(texts) {
    return texts.map((text) =>
      text.split(" ").map((word) => this.wordIndex[word] || 0),
    );
  }
}

function padSequences(sequences, maxlen) {
  return sequences.map((seq) =>
    seq.length < maxlen
      ? [...Array(maxlen - seq.length).fill(0), ...seq]
      : seq.slice(0, maxlen),
  );
}

async function predictCategory() {
  const text = document.getElementById("textInput").value;
  if (!text.trim()) {
    document.getElementById("result").innerText =
      "Please enter text to classify.";
    return;
  }

  try {
    const tokenizedText = tokenizer.textsToSequences([text]);
    const paddedText = padSequences(tokenizedText, 100);
    const inputTensor = tf.tensor2d(paddedText);
    const prediction = model.predict(inputTensor);
    const predictedClass = prediction.argMax(1).dataSync()[0];
    const predictedCategory = categoryNames[predictedClass];
    document.getElementById("result").innerText =
      `Predicted Category: ${predictedCategory}`;
  } catch (error) {
    document.getElementById("result").innerText =
      "Error: Could not make a prediction. Check the console.";
  }
}

window.onload = loadModelAndTokenizer;
