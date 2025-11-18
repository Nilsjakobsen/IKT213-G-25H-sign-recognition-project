# 🚦 Traffic Sign Recognition Project

A comprehensive traffic sign recognition system that extracts and classifies Norwegian road signs from PDF construction plans using computer vision and deep learning.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Web Application](#web-application)
- [System Components](#system-components)
- [Output](#output)
- [Contributing](#contributing)

## 🎯 Overview

This project processes PDF construction plans (APV plans) to automatically detect and classify traffic signs. It uses a three-stage pipeline:

1. **Map Extraction** - Converts PDF pages to images and filters out duplicate maps
2. **Sign Extraction** - Detects and extracts individual traffic signs using color-based segmentation
3. **Sign Classification** - Classifies extracted signs using a trained Convolutional Neural Network

The system includes both a command-line interface and a modern web application with a React frontend and Flask backend.

## ✨ Features

- 📄 PDF to image conversion with duplicate map detection using ORB feature matching
- 🔍 Color-based sign detection (red borders, yellow/white centers)
- 🧠 CNN-based sign classification supporting 11+ Norwegian traffic sign classes
- 🌐 Web interface for easy PDF upload and result visualization
- 📊 CSV export of classification results with confidence scores
- 🎨 Data augmentation tools for training dataset expansion

## 📁 Project Structure

```
├── main.py                          # Main pipeline script
├── README.md                        # This file
├── Sign_processing/                 # Core processing modules
│   ├── Map_extractor.py            # PDF to image conversion & duplicate detection
│   ├── Sign_extractor.py           # Sign detection and extraction
│   ├── cnn.py                      # CNN model definition and predictor
│   ├── cnnTrainer.py               # Model training utilities
│   ├── augmentations.py            # Data augmentation functions
│   └── demo/                       # Pre-trained model and classes
│       ├── cnn.pth                 # Trained model weights
│       ├── classes.json            # Sign class labels
│       └── train/val/              # Training/validation datasets
├── frontend_files/                  # Web application
│   ├── backend/
│   │   ├── app.py                  # Flask API server
│   │   └── requirements-backend.txt # Backend dependencies
│   ├── frontend/                   # React application
│   │   ├── src/
│   │   │   ├── App.jsx            # Main React component
│   │   │   └── components/        # UI components
│   │   ├── package.json           # Node.js dependencies
│   │   └── vite.config.js         # Build configuration
│   ├── scripts/                    # Helper scripts
│   │   ├── start-dev.sh           # Development startup (Linux/Mac)
│   │   ├── start-dev.bat          # Development startup (Windows)
│   │   └── check_installation.py  # Verify installation
│   └── docs/                       # Additional documentation
├── Outputs/                         # Processing outputs
│   ├── Output_from_Map_extractor/  # Extracted map images
│   ├── Output_from_Sign_extractor/ # Detected sign images
│   └── sign_predictions.csv        # Classification results
├── temp_uploads/                    # Temporary upload storage
└── APV_plan_GDPR_trygg/            # Sample input PDFs

```

## 📦 Requirements

### System Requirements

- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **Conda**: Recommended for environment management
- **Node.js**: 16.x or higher (for web frontend)
- **npm**: 7.x or higher

### Python Dependencies

#### Core Processing Libraries
```
opencv-python>=4.8.0
opencv-contrib-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
pytesseract>=0.3.10
pdf2image>=1.16.0
torch>=2.0.0
torchvision>=0.15.0
```

#### Web Backend
```
Flask==3.0.0
flask-cors==4.0.0
Werkzeug==3.0.1
```

#### Additional System Dependencies
- **Tesseract OCR**: Required for text detection
- **Poppler**: Required for PDF to image conversion

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Nilsjakobsen/IKT213-G-25H-sign-recognition-project.git
cd IKT213-G-25H-sign-recognition-project
```

### Step 2: Set Up Python Environment

#### Using Conda (Recommended)

```bash
# Create and activate conda environment
conda create -n ikt213 python=3.10
conda activate ikt213
```

#### Using venv (Alternative)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install System Dependencies

#### On Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils
```

#### On macOS

```bash
brew install tesseract poppler
```

#### On Windows

1. Download and install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
2. Download and install [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/)
3. Add both to your system PATH

### Step 4: Install Python Dependencies

#### For Core Processing Only

```bash
pip install opencv-python opencv-contrib-python numpy Pillow pytesseract pdf2image torch torchvision
```

#### For Web Application (Backend)

```bash
cd frontend_files/backend
pip install -r requirements-backend.txt
cd ../..
```

### Step 5: Install Node.js Dependencies (For Web Interface)

```bash
cd frontend_files/frontend
npm install
cd ../..
```

### Step 6: Verify Installation

```bash
python frontend_files/scripts/check_installation.py
```

## 🎮 Usage

### Command Line Interface

Run the complete pipeline on a PDF file:

```bash
python main.py
```

The script will:
1. Convert PDF pages to images
2. Extract traffic signs from each page
3. Classify each detected sign using the CNN
4. Save results to `Outputs/sign_predictions.csv`

### Customizing Input

Edit `main.py` to specify your PDF file:

```python
APV_plan_GDPR_trygg = "path/to/your/plan.pdf"
```

### Output Files

- **Maps**: `Outputs/Output_from_Map_extractor/page_*.jpg`
- **Signs**: `Outputs/Output_from_Sign_extractor/*.png`
- **Results**: `Outputs/sign_predictions.csv`

## 🌐 Web Application

### Starting the Development Server

#### Linux/macOS

```bash
cd frontend_files/scripts
chmod +x start-dev.sh
./start-dev.sh
```

#### Windows

```cmd
cd frontend_files\scripts
start-dev.bat
```

### Manual Startup

#### Terminal 1: Backend Server

```bash
conda activate ikt213
cd frontend_files/backend
python app.py
```

The Flask API will run on `http://localhost:5000`

#### Terminal 2: Frontend Server

```bash
cd frontend_files/frontend
npm run dev
```

The React app will run on `http://localhost:3000`

### Using the Web Interface

1. Open `http://localhost:3000` in your browser
2. Upload a PDF construction plan
3. Wait for processing (progress indicator shown)
4. View detected signs with classification results
5. Review confidence scores for each prediction

### API Endpoints

- `POST /api/upload` - Upload and process PDF file
- `GET /api/results/<session_id>` - Retrieve processing results
- `GET /api/sign/<session_id>/<filename>` - Get individual sign image

For detailed API documentation, see `frontend_files/docs/API_EXAMPLES.md`

## 🔧 System Components

### Map Extractor (`Map_extractor.py`)

- Converts PDF pages to JPEG images using `pdf2image`
- Uses ORB (Oriented FAST and Rotated BRIEF) feature matching to detect duplicate maps
- Configurable parameters:
  - `nfeatures`: Number of ORB features (default: 20000)
  - `ratio`: Distance ratio threshold (default: 0.75)
  - `min_good`: Minimum good matches for duplicate detection (default: 12000)

### Sign Extractor (`Sign_extractor.py`)

- **HSV Color Detection**: Identifies red borders and yellow/white centers
- **Contour Analysis**: Finds sign boundaries using morphological operations
- **Shape Filtering**: Validates signs based on aspect ratio and area
- **ROI Extraction**: Crops and saves individual signs with margins

### CNN Classifier (`cnn.py`)

- **Architecture**: 4 convolutional layers + 2 fully connected layers
- **Input**: 128x128 RGB images
- **Output**: 11+ traffic sign classes with confidence scores
- **Features**:
  - Dropout regularization (0.3)
  - Max pooling
  - ReLU activation
  - Softmax classification

### Supported Sign Classes

- Direction/warning: 110, 132g, 149g, 150m
- Combined signs: 110+132g
- Speed limits: 362_50g, 362_60, 362_60g, 362_70, 362_80
- Special: manuellDirigering, vegmarkeringMangler

## 📊 Output

### CSV Results Format

```csv
filename,predicted_class,confidence
sign_0001.png,110,0.9823
sign_0002.png,362_60,0.8654
sign_0003.png,manuellDirigering,0.9201
```

### File Naming Convention

- **Maps**: `page_001.jpg`, `page_002.jpg`, etc.
- **Signs**: `sign_0001.png`, `sign_0002.png`, etc.

## 🔄 Training Your Own Model

### Step 1: Generate Training Data with Augmentation

Expand your training dataset using data augmentation:

```bash
python Sign_processing/augmentations.py
```

This will apply various transformations (rotation, scaling, brightness adjustments, etc.) to your existing training images, creating more diverse training samples.

### Step 2: Train the CNN Model

Train the model using the augmented dataset:

```bash
python3 -m Sign_processing.cnnTrainer train
```

This command will:
- Load training and validation data from `Sign_processing/demo/train` and `Sign_processing/demo/val`
- Train the CNN with class balancing and weighted sampling
- Save the trained model to `Sign_processing/demo/cnn.pth`
- Save class labels to `Sign_processing/demo/classes.json`

### Advanced Training Options

For programmatic training with custom parameters:

```python
from Sign_processing.cnnTrainer import CNNTrainer

trainer = CNNTrainer(
    train_dir="Sign_processing/demo/train",
    val_dir="Sign_processing/demo/val",
    model_path="Sign_processing/demo/cnn.pth",
    classes_path="Sign_processing/demo/classes.json"
)

trainer.train(epochs=50, batch_size=32)
```

See `cnnTrainer.py` for advanced training options including class weighting and learning rate scheduling.

## 🛠️ Troubleshooting

### Common Issues

**1. Tesseract not found**
```bash
# Set the tesseract path explicitly in your code
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
```

**2. Poppler not found**
```bash
# On Windows, specify poppler path
convert_from_path(pdf_path, poppler_path=r'C:\path\to\poppler\bin')
```

**3. CUDA/GPU issues**
```python
# Force CPU mode
device = torch.device('cpu')
```

**4. Port already in use**
```bash
# Kill process on port 5000 (Linux/Mac)
lsof -ti:5000 | xargs kill -9

# Kill process on port 5000 (Windows)
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

For more troubleshooting help, see `frontend_files/docs/TROUBLESHOOTING_ERRNO5.md`

## 📚 Additional Documentation

- **Quick Start**: `frontend_files/docs/QUICKSTART.md`
- **Frontend Guide**: `frontend_files/docs/FRONTEND_README.md`
- **Architecture**: `frontend_files/docs/ARCHITECTURE.md`
- **API Examples**: `frontend_files/docs/API_EXAMPLES.md`
- **Installation Help**: `frontend_files/docs/INSTALL_GUIDE.md`

## 👥 Contributing

This is a course project (IKT213-G-25H) developed for sign recognition from construction plans.

### Team
- Repository: [Nilsjakobsen/IKT213-G-25H-sign-recognition-project](https://github.com/Nilsjakobsen/IKT213-G-25H-sign-recognition-project)

### Course Information
- **Course**: IKT213
- **Institution**: University of Agder (UiA)
- **Year**: 2025

## 📄 License

This project is created for educational purposes as part of the IKT213 course.

## 🙏 Acknowledgments

- PDF processing: `pdf2image` and `Poppler`
- Computer vision: OpenCV
- Deep learning: PyTorch
- OCR: Tesseract
- Web framework: Flask and React
- Build tool: Vite

---

**Last Updated**: November 2025
**Version**: 1.0
