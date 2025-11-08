

🫁 X-Ray Disease Classifier App
This is a Streamlit application that uses a pre-trained deep learning model (DenseNet-121 from the torchxrayvision library) to perform multi-label classification on uploaded chest X-ray images. It provides confidence scores for various thoracic pathologies and offers explainability features like a Saliency Map.

✨ Features
Image Upload: Accepts PNG/JPG chest X-ray images.

AI Diagnosis: Classifies up to 19 different thoracic pathologies (based on NIH/ChestX-ray8 dataset).

Normalcy Score: Provides an overall confidence score for a normal study.

Post-hoc Correction: Adjusts disease probabilities downward if the Normalcy Score is high.

Interactive Report: Displays top findings and detailed descriptions based on a user-defined probability threshold.

Explainability: Generates a Saliency Map to highlight areas of the image the model focused on for its top prediction.

⚙️ Setup and Installation
1. Prerequisites
You need Python (3.8+) installed on your system.

2. Clone the Repository
Since you are running this locally, ensure your project folder (ChestXrayPrototype1) contains the file app-test.py.

3. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies cleanly.

Bash

# Create the environment (e.g., named 'venv')
python -m venv venv

# Activate the environment
# On Windows (Command Prompt/PowerShell):
.\venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
4. Install Dependencies
You need to install Streamlit, PyTorch, Torchvision, TorchXRayVision, and other dependencies.

Bash

pip install streamlit torch torchvision numpy pandas pillow scikit-image torchxrayvision
Note: PyTorch (torch) installation might be complex depending on your system (CPU vs. GPU). The current script is configured to use the CPU, simplifying the installation.

🚀 How to Run the App (VS Code)
Visual Studio Code (VS Code) is the recommended environment for Python development and integrates perfectly with Streamlit.

1. Open the Project in VS Code
Open VS Code.

Go to File > Open Folder...

Select your project directory (C:\Users\Asus\Downloads\ChestXrayPrototype1).

2. Configure the Python Interpreter
Open the Command Palette (Ctrl+Shift+P or Cmd+Shift+P).

Type "Python: Select Interpreter".

Select the virtual environment you created (e.g., .\venv\Scripts\python.exe). You should see (venv) appear on the left side of your VS Code status bar.

3. Run the Streamlit App
Open a new VS Code Terminal (Ctrl+** or **Cmd+). This terminal will automatically use the activated (venv).

Run the application using the Streamlit command:

Bash

streamlit run app-test.py
The app will launch in your default web browser (e.g., at http://localhost:8502).

To stop the app, go back to the terminal and press Ctrl+C.

📋 File Structure
ChestXrayPrototype1/
├── app-test.py        # The main Streamlit application file
└── README.md          # This file
└── venv/              # Python Virtual Environment
⚠️ Important Disclaimer
This is an AI-assisted research tool for demonstrating multi-label classification. It is not a replacement for professional medical interpretation. Always consult qualified healthcare providers for diagnosis and treatment.
