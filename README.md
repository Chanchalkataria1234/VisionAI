# VisionAI — CNN Image Classifier

VisionAI is a client-side deep learning application that uses a trained Convolutional Neural Network (CNN) in the browser to identify images of **aeroplanes**, **cars**, and **birds**. Running entirely in the browser using TensorFlow.js, it provides instant predictions without uploading any data to external servers, ensuring 100% user privacy and fast execution.

---

## ⚡ Features
- **In-Browser Inference:** Powered by [TensorFlow.js](https://js.tensorflow.org/), executing model predictions client-side.
- **Modern UI/UX:** Built with sleek aesthetics, featuring a deep blue dark mode layout, smooth neon glow animations, drag-and-drop uploads, and dynamic confidence score bars.
- **Privacy-First:** Since predictions happen locally on your GPU/CPU, your images never leave your machine.
- **Custom Trained CNN:** Built using Keras, trained on a tailored subset of the CIFAR-10 dataset (Aeroplane, Car, Bird).

---

## 📁 Project Structure
The repository is organized as follows:
- **[index.html](file:///c:/Users/Chanchal%20Kataria/Desktop/Image_Classification/index.html)**: The web application markup and layout.
- **[app.js](file:///c:/Users/Chanchal%20Kataria/Desktop/Image_Classification/app.js)**: Core client-side JavaScript containing TF.js model loading, image pre-processing, and interactive UI animations.
- **[style.css](file:///c:/Users/Chanchal%20Kataria/Desktop/Image_Classification/style.css)**: CSS stylesheet with responsive grid design, interactive hover effects, and modern styling cues.
- **[Student Notebook.ipynb](file:///c:/Users/Chanchal%20Kataria/Desktop/Image_Classification/Student%20Notebook.ipynb)**: Jupyter Notebook detailing the model training pipeline (subsetting CIFAR-10, training the CNN, compiling, and evaluating).
- **[convert_model.py](file:///c:/Users/Chanchal%20Kataria/Desktop/Image_Classification/convert_model.py)**: Python script to load the trained Keras HDF5 model weights and serialize them to web-compatible TensorFlow.js layers format.
- **`models/`**: Stores training checkpoints (`.keras` format) and converted HDF5 models (`model.h5`, `model_keras2.h5`).
- **`web_model/`**: Converted model files (`model.json` and binary weight shards) loaded directly by `app.js`.

---

## 🧠 Model Architecture & Training
The classifier is a customized Convolutional Neural Network (CNN) trained on **15,000 training samples** and validated against **3,000 test samples** from the CIFAR-10 dataset.

### Architecture Details:
1. **Input Layer:** `32x32` resolution with 3 RGB color channels.
2. **Convolutional Block 1:**
   - 2D Convolution (32 filters, 3x3 kernel, ReLU activation, padding=same)
   - Batch Normalization
   - 2D Convolution (32 filters, 3x3 kernel, ReLU activation)
   - Max Pooling (2x2 pool size)
   - Dropout (50% rate)
3. **Convolutional Block 2:**
   - 2D Convolution (64 filters, 3x3 kernel, ReLU activation, padding=same)
   - Batch Normalization
   - 2D Convolution (64 filters, 3x3 kernel, ReLU activation)
   - Max Pooling (2x2 pool size)
   - Dropout (50% rate)
4. **Convolutional Block 3:**
   - 2D Convolution (128 filters, 3x3 kernel, ReLU activation, padding=same)
   - Batch Normalization
   - 2D Convolution (128 filters, 3x3 kernel, ReLU activation)
   - Max Pooling (2x2 pool size)
   - Dropout (50% rate)
5. **Classification Layer:**
   - Flatten
   - Dense Fully Connected (3 nodes, Softmax activation)

---

## 🚀 Running the Web App Locally

Because of browser Security Policies (specifically **CORS** restrictions), loading local files (such as `web_model/model.json`) via the standard `file:///` protocol will fail. You must run a local HTTP server to host the files.

### Option 1: Python HTTP Server (Recommended)
If you have Python installed, open a terminal in the project directory and run:
```bash
python -m http.server 8000
```
Then visit **`http://localhost:8000`** in your web browser.

### Option 2: Live Server Extension (VS Code)
If you use VS Code, install the **Live Server** extension, open the project workspace, and click the **"Go Live"** button in the bottom status bar.

---

## 🔄 Re-converting the Keras Model (Optional)

If you modify the training code in `Student Notebook.ipynb` and export a new model to `models/model.h5`, you can rebuild and save the web model using the conversion script.

### Requirements:
Install the required packages in your Python environment:
```bash
pip install tensorflow tensorflowjs h5py numpy
```

### Run the Converter:
```bash
python convert_model.py
```
This script will recreate the Keras 2 architecture, load the weights from `models/model.h5`, output a Keras 2 compatible `.h5` file, and invoke `tfjs.converters.save_keras_model` to save the converted weights directly to the `web_model/` directory.

---

## 📝 Image Preprocessing & Inference Details

During inference, images uploaded through the UI are preprocessed inside `app.js` using TensorFlow.js operations to exactly match the network's training input:
1. **Pixel Retrieval:** Reading raw image data using `tf.browser.fromPixels`.
2. **Resizing:** Bilinear resizing from the original resolution to `32x32` pixels (`tf.image.resizeBilinear`).
3. **Normalization:** Scaling color channel values from `[0, 255]` to `[0.0, 1.0]` by dividing by `255.0` (`tf.cast(..., 'float32').div(255.0)`).
4. **Batching:** Expanding the dimensions to insert a batch size of `1`, outputting a tensor shape of `[1, 32, 32, 3]`.
5. **Prediction:** Running `model.predict` to output an array of 3 probabilities, which are mapped to their respective classes (Aeroplane, Car, Bird).

---

## 🛠️ Technology Stack
- **Frontend Core:** HTML5, Vanilla JavaScript, CSS3 variables.
- **Model Framework:** TensorFlow 2 / Keras (Python) for training.
- **Web Inference:** TensorFlow.js CDN (`@tensorflow/tfjs@4.17.0`) for browser-based runtime.
