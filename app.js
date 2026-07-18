// app.js

const CLASS_NAMES = ["Aeroplane", "Car", "Bird"];
const CLASS_ICONS = {
    "Aeroplane": "✈️",
    "Car": "🚗",
    "Bird": "🐦"
};

// UI Elements
const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const uploadPlaceholder = document.getElementById('upload-content-placeholder');
const previewContainer = document.getElementById('preview-container');
const imagePreview = document.getElementById('image-preview');

const analyzeBtn = document.getElementById('analyze-btn');
const clearBtn = document.getElementById('clear-btn');

const resultsEmptyState = document.getElementById('results-empty-state');
const resultsLoadingState = document.getElementById('results-loading-state');
const resultsOutputContainer = document.getElementById('results-output-container');

const badgeIcon = document.getElementById('badge-icon');
const predictedClassName = document.getElementById('predicted-class-name');
const predictedConfidence = document.getElementById('predicted-confidence');

const pctAeroplane = document.getElementById('pct-aeroplane');
const fillAeroplane = document.getElementById('fill-aeroplane');
const pctCar = document.getElementById('pct-car');
const fillCar = document.getElementById('fill-car');
const pctBird = document.getElementById('pct-bird');
const fillBird = document.getElementById('fill-bird');

let model = null;
let uploadedImage = null;

// Initialize: Asynchronously load TensorFlow.js model
async function loadModel() {
    try {
        console.log("Loading TensorFlow.js model...");
        // Look for model.json inside the web_model directory relative to the page
        model = await tf.loadLayersModel('web_model/model.json');
        console.log("TensorFlow.js model loaded successfully!", model);
    } catch (err) {
        console.error("Error loading the model:", err);
        
        // Show a helpful warning if loading failed (commonly due to CORS when using file:// protocol)
        if (window.location.protocol === 'file:') {
            alert("Security Warning: Browsers block loading local files (CORS) when running via 'file:///' URLs. \n\nPlease run a local HTTP server to test. Open your terminal and run:\npython -m http.server 8000\nThen visit http://localhost:8000 in your browser.");
        } else {
            alert("Failed to load the model. Please ensure the 'web_model' folder exists in the project root containing model.json and weight files.");
        }
    }
}

// Set up event listeners
function setupEventListeners() {
    // Upload zone click triggers file input click
    uploadZone.addEventListener('click', () => {
        fileInput.click();
    });

    // File selection event
    fileInput.addEventListener('change', (e) => {
        handleFileSelect(e.target.files[0]);
    });

    // Drag and drop event handlers
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = 'rgba(34, 211, 238, 0.8)';
        uploadZone.style.backgroundColor = '#0e1b33';
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.style.borderColor = 'rgba(129, 140, 248, 0.35)';
        uploadZone.style.backgroundColor = '#0b1325';
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = 'rgba(129, 140, 248, 0.35)';
        uploadZone.style.backgroundColor = '#0b1325';
        
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    // Analyze button click
    analyzeBtn.addEventListener('click', () => {
        if (uploadedImage && model) {
            analyzeImage();
        }
    });

    // Clear button click
    clearBtn.addEventListener('click', () => {
        resetApp();
    });
}

// Handle selected image file
function handleFileSelect(file) {
    if (!file || !file.type.startsWith('image/')) {
        alert("Please select a valid image file.");
        return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
        imagePreview.src = event.target.result;
        
        // Hide placeholder and show preview
        uploadPlaceholder.style.display = 'none';
        previewContainer.style.display = 'flex';
        
        uploadedImage = new Image();
        uploadedImage.src = event.target.result;
        uploadedImage.onload = () => {
            // Enable analyze button once image is loaded
            analyzeBtn.disabled = false;
        };
    };
    reader.readAsDataURL(file);
}

// Reset application UI state
function resetApp() {
    fileInput.value = '';
    uploadedImage = null;
    imagePreview.src = '';
    
    // Restore upload zone elements
    uploadPlaceholder.style.display = 'block';
    previewContainer.style.display = 'none';
    
    // Disable analyze button
    analyzeBtn.disabled = true;
    
    // Restore results empty state
    resultsEmptyState.style.display = 'flex';
    resultsLoadingState.style.display = 'none';
    resultsOutputContainer.style.display = 'none';
    
    // Reset progress fills
    fillAeroplane.style.width = '0%';
    fillCar.style.width = '0%';
    fillBird.style.width = '0%';
    
    pctAeroplane.innerText = '0%';
    pctCar.innerText = '0%';
    pctBird.innerText = '0%';
}

// Main Image Inference
async function analyzeImage() {
    if (!model) {
        alert("Model is not loaded yet. Please refresh the page or check the console.");
        return;
    }

    // Toggle loading states
    resultsEmptyState.style.display = 'none';
    resultsLoadingState.style.display = 'flex';
    resultsOutputContainer.style.display = 'none';

    // Wait slightly to let the loading animation render
    await new Promise(resolve => setTimeout(resolve, 300));

    try {
        // Preprocess image tensor using tfjs
        const logits = tf.tidy(() => {
            // Read image pixels
            const imgTensor = tf.browser.fromPixels(uploadedImage);
            
            // Resize to 32x32 matching CIFAR-10 model input shape
            const resizedTensor = tf.image.resizeBilinear(imgTensor, [32, 32]);
            
            // Cast to float and scale values to [0, 1] range (same as x_train / 255.)
            const normalizedTensor = tf.cast(resizedTensor, 'float32').div(tf.scalar(255.0));
            
            // Expand dimension to insert batch size [1, 32, 32, 3]
            const batchedTensor = normalizedTensor.expandDims(0);
            
            // Predict
            return model.predict(batchedTensor);
        });

        // Retrieve raw probabilities
        const probabilities = await logits.data();
        logits.dispose(); // clean up memory

        console.log("Prediction probabilities:", probabilities);

        // Find the index of the class with highest probability
        let maxIndex = 0;
        let maxVal = -1;
        for (let i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > maxVal) {
                maxVal = probabilities[i];
                maxIndex = i;
            }
        }

        const predictedClass = CLASS_NAMES[maxIndex];
        const confidencePct = (maxVal * 100).toFixed(2);

        // Update main result badge
        badgeIcon.innerText = CLASS_ICONS[predictedClass];
        predictedClassName.innerText = predictedClass;
        predictedConfidence.innerText = `${confidencePct}% confidence`;

        // Update probability values
        const probAeroplane = (probabilities[0] * 100).toFixed(1);
        const probCar = (probabilities[1] * 100).toFixed(1);
        const probBird = (probabilities[2] * 100).toFixed(1);

        pctAeroplane.innerText = `${probAeroplane}%`;
        pctCar.innerText = `${probCar}%`;
        pctBird.innerText = `${probBird}%`;

        // Hide loading spinner and display results container
        resultsLoadingState.style.display = 'none';
        resultsOutputContainer.style.display = 'flex';

        // Trigger animations for the progress bars
        setTimeout(() => {
            fillAeroplane.style.width = `${probAeroplane}%`;
            fillCar.style.width = `${probCar}%`;
            fillBird.style.width = `${probBird}%`;
        }, 100);

    } catch (err) {
        console.error("Error during image analysis:", err);
        alert("An error occurred during image inference. Check the console logs for details.");
        resetApp();
    }
}

// Start app
loadModel();
setupEventListeners();
