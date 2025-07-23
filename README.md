# Landmark Image Classification: From Scratch & Transfer Learning with PyTorch 🖼️

This project explores landmark image classification using two distinct deep learning approaches in PyTorch. It covers the full lifecycle from data preparation and model training (custom CNN and ResNet18 transfer learning) to deployment as a **containerized FastAPI application using Docker**. An interactive Jupyter Notebook for experimentation is also included.

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.70%2B-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-20.10%2B-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

*   [Key Features](#key-features)
*   [Deployment with Docker & FastAPI](#deployment-with-docker--fastapi)
    *   [Prerequisites](#prerequisites)
    *   [Running the Pre-built Image](#running-the-pre-built-image)
    *   [Building the Image Locally (Optional)](#building-the-image-locally-optional)
    *   [Interacting with the API](#interacting-with-the-api)
*   [Local Development & Training](#local-development--training)
    *   [Project Structure](#project-structure)
    *   [Setup & Installation](#setup--installation)
    *   [Usage](#usage)
*   [Model Details](#model-details)
*   [Technology Stack](#technology-stack)
*   [Contributing](#contributing)
*   [License](#license)

---

## Key Features ✨

*   **API Deployment:** The trained model is served via a **FastAPI** application, containerized with **Docker** for easy deployment.
*   **Dual Approach:** Implements and compares a custom CNN built from scratch against a fine-tuned pre-trained ResNet18 model.
*   **End-to-End Pipeline:** Covers data loading, extensive augmentation (RandAugment, etc.), training, validation, and testing.
*   **Model Export:** Demonstrates model serialization using `torch.jit.script` for efficient, deployment-ready models.
*   **Interactive Demo:** Includes a Jupyter Notebook (`app.ipynb`) with `ipywidgets` for rapid experimentation.
*   **Unit Tested:** Core functionalities are backed by `pytest` unit tests to ensure reliability.
*   **Modular Code:** Well-organized `src/` directory for reusable components.

---

## Deployment with Docker & FastAPI 🐳

The easiest way to run the prediction service is using the pre-built Docker image available on Docker Hub.

### Prerequisites
*   [Docker](https://www.docker.com/get-started) installed and running on your system.

### Running the Pre-built Image

Pull and run the container in one command. This will expose the API on `port 80` of your local machine.

```bash
docker run -p 80:80 --name landmark-api skullgrinder123/landmark-api:1.0
```
The API will now be accessible at http://localhost:80

## Interacting with the API
You can send a POST request with an image to the /predict/ endpoint to get a classification. Use one of the images from the test/ directory or your own.
Here's an example using curl:
```bash
curl -X POST -F "file=@test/image1.jpg" http://localhost/predict/
```
## Expected Response:

```json
{
  "filename": "image1.jpg",
  "prediction": "CLASS_NAME"
}
```
You can also visit http://localhost/docs in your browser to see the interactive Swagger UI documentation for the API (Amazing feature in FastAPI).

## Project Structure 📂

```
landmark-classification-pytorch/
├── api/                   # FastAPI application for deployment
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt # dependencies inside the container
├── static_images/icons/                 # Icons used in Jupyter notebooks
├── landmark_images/       # Dataset (downloaded automatically when you run the script)
│   ├── train/
│   └── test/
├── Models/                # Exported .pt models for inference
│   ├── original_exported.pt
│   └── transfer_exported.pt
├── src/                   # Source code for the project
│   ├── __init__.py
│   ├── data.py            # Data loading, augmentation, and preprocessing
│   ├── helpers.py         # Utility functions (dataset download, mean/std calc)
│   ├── model.py           # Custom CNN architecture
│   ├── optimization.py    # Loss function and optimizer setup
│   ├── predictor.py       # Wrapper class for inference with exported models
│   ├── train.py           # Training, validation, and testing loops
│   └── transfer.py        # Transfer learning model setup
├── test/                  # Sample images for testing the application
│   ├── image1.jpg
│   └── image2.png
├── app.ipynb              # Interactive landmark classification application
├── cnn_from_scratch.ipynb # Notebook for training the custom CNN
├── transfer_learning.ipynb# Notebook for training with transfer learning
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## Setup & Installation ⚙️

### Prerequisites

*   Python (3.8 or higher recommended)
*   `pip` and `virtualenv` (recommended)
*   Git

### Cloning the Repository

```bash
git clone https://github.com/BinarySkull/Landmark-Image-Classification.git
```

### Setting up the Environment

It's highly recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```
If you encounter issues with `jupyter nbconvert` for HTML generation (used in `create_submit_pkg.py`), you might need to install Pandoc and TeX (e.g., MiKTeX on Windows, MacTeX on macOS, or `texlive` on Linux).

### Dataset

The landmark image dataset will be automatically downloaded and extracted by the helper scripts (`src/helpers.py`) when first needed (e.g., when running `src.data.get_data_loaders`). It will be placed in a folder named `landmark_images/` in the project root.

If you have the dataset (`landmark_images.zip`) manually, you can extract it to the project root directory so that you have `landmark_images/train` and `landmark_images/test`.

---

## Usage 🚀

### 1. Data Preparation & Exploration

The helper script `src/helpers.py` contains a function `setup_env()` which handles:
*   Setting random seeds for reproducibility.
*   Downloading and extracting the dataset if not present.
*   Computing (or loading cached) mean and standard deviation of the dataset for normalization.
*   Creating a `checkpoints/` directory.

This function is often called implicitly by other scripts or notebooks when data operations are performed. You can explore data visualization in the initial cells of `cnn_from_scratch.ipynb` or `transfer_learning.ipynb`.

### 2. Training the CNN from Scratch

Open and run the cells in `cnn_from_scratch.ipynb`:
*   This notebook defines hyperparameters for training.
*   It uses `src.model.MyModel` for the custom CNN architecture.
*   Training progress is visualized, and the best model weights are saved to `checkpoints/best_val_loss.pt`.
*   The trained model is then evaluated on the test set.
*   Finally, it exports the trained custom model to `models/original_exported.pt`.

### 3. Training with Transfer Learning

Open and run the cells in `transfer_learning.ipynb`:
*   This notebook configures hyperparameters for transfer learning (e.g., using ResNet18).
*   It utilizes `src.transfer.get_model_transfer_learning` to load a pre-trained model and adapt its classifier.
*   The training process saves the best model to `checkpoints/model_transfer.pt`.
*   The fine-tuned model is evaluated, and a confusion matrix can be plotted.
*   The model is exported to `models/transfer_exported.pt` (or `checkpoints/transfer_exported.pt` as per the notebook's code, ensure consistency).

### 4. Running the Interactive Classification App

The `app.ipynb` notebook provides an interactive way to classify landmark images.

1.  **Open `app.ipynb` in Jupyter Notebook or JupyterLab.**
2.  **Choose a Model:** By default, the app is configured to load one of the exported models (e.g., `models/transfer_exported.pt`). You can modify the cell:
    ```python
    # Decide which model you want to use among the ones exported
    learn_inf = torch.jit.load("models/transfer_exported.pt") # Or "models/original_exported.pt"
    ```
3.  **Run the Cells:** Execute all cells in the notebook.
4.  **Upload an Image:**
    *   Click the "Upload" button.
    *   Select an image file. You can use images from the `test/` directory or any other landmark image.
5.  **Classify:**
    *   Click the "Classify" button.
    *   The app will display the uploaded image and the top 5 predicted landmark classes with their probabilities.

### 5. Running Tests

To run the unit tests using `pytest` (ensure it's installed via `requirements.txt`):

```bash
pytest
```

This will execute tests defined in the `src/` Python files.

---

## Model Details 🧠

### Custom CNN

The custom CNN (`src/model.py`) is built with a sequence of:
*   Convolutional layers (`nn.Conv2d`) with increasing filter depth.
*   Batch Normalization (`nn.BatchNorm2d`) for stable training.
*   ReLU activation functions (`nn.ReLU`).
*   Dropout layers (`nn.Dropout2d`) for regularization.
*   Max Pooling layers (`nn.MaxPool2d`) for downsampling.
*   An Adaptive Average Pooling layer (`nn.AdaptiveAvgPool2d`) to handle variable input sizes before the classifier.
*   Fully connected layers (`nn.Linear`) for classification, also with Batch Normalization and Dropout.
*   Xavier Uniform initialization for weights.

### Transfer Learning (ResNet18)

The transfer learning approach (`src/transfer.py`):
*   Loads a pre-trained ResNet18 model from `torchvision.models` with `pretrained=True`.
*   Freezes all parameters of the convolutional base to retain learned features.
*   Replaces the final fully connected layer (`fc`) with a new `nn.Linear` layer tailored to the number of landmark classes in our dataset. Only this new layer is trained initially.

---

## Technologies used 🛠️

*   **Python:** Core programming language.
*   **PyTorch:** Deep learning framework for model building, training, and inference.
*   **TorchVision:** For pre-trained models (ResNet18) and common image transformations.
*   **NumPy:** For numerical operations.
*   **Matplotlib & Seaborn:** For plotting and visualizations (e.g., training curves, confusion matrix).
*   **Pillow (PIL):** For image manipulation in the interactive app.
*   **ipywidgets & IPython:** For creating the interactive GUI in Jupyter Notebook.
*   **pytest:** For unit testing.
*   **tqdm:** For progress bars.
*   **LiveLossPlot:** For real-time plotting of training metrics.
*   **Jupyter Notebook/Lab:** For interactive development and demonstration.
*   **Docker:** For containerizing the endpoint, creating the image, and pushing it to Docker Hub.
*   **FastAPI:** For creating endpoints to use the model.

---

## Contributing 🤝

Contributions, issues, and feature requests are welcome! Please feel free to fork the repository, make changes, and open a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## License 📄

This project is licensed under the MIT License - Have Fun.

---

Happy Classifying! If you have any questions, feel free to open an issue.
