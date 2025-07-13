# Face Emotion Recognition using CNN

## Overview

This project implements a Convolutional Neural Network (CNN) for real-time facial emotion recognition. The model can detect and classify seven different emotions from facial expressions:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

The system can analyze emotions from both webcam live feeds and video files with an accuracy of approximately 70% on the FER2013 dataset.

## Features

- Real-time emotion detection from webcam feed
- Emotion detection from video files
- Confidence threshold for emotion prediction
- Frame rate display
- Visual bounding box around detected faces
- User-friendly interface
- Performance optimization with multi-threading
- GPU acceleration support
- Parallel face processing

## Dataset

The model was trained on the FER2013 dataset, which contains 48x48 pixel grayscale images of faces categorized into seven emotions. The dataset is split into training and testing directories, each containing subdirectories for the seven emotion categories.

## Model Architecture

The CNN model architecture consists of:

- Multiple convolutional layers with batch normalization and max pooling
- Dropout layers to prevent overfitting
- Dense layers for classification
- Adam optimizer with learning rate reduction
- Categorical cross-entropy loss function

## Requirements

All required packages are listed in the `requirements.txt` file. Key dependencies include:

- TensorFlow and Keras for deep learning
- OpenCV for image processing and face detection
- NumPy for numerical operations
- PIL for image manipulation
- Pandas for data handling

## Installation

1. Clone the repository:

```
git clone https://github.com/yourusername/Face-Emotion-Recognition-in-Python-using-CNN.git
cd Face-Emotion-Recognition-in-Python-using-CNN
```

2. Install the required packages:

```
pip install -r requirements.txt
```

## Usage

### Real-time Detection

To run emotion detection on a live webcam feed:

```
python start.py
```

Press 'q' to exit the application.

### Video File Analysis

To run emotion detection on a video file:

```
python start.py --video <video_name>
```

### Training Your Own Model

The project includes Jupyter notebooks for training:

- `trainmodel.ipynb`: Contains the code for training and saving the model
- `model.ipynb`: Contains model evaluation and analysis

## Model Files

- `emotiondetector.json`: Contains the model architecture
- `emotiondetector.h5`: Contains the trained model weights

## Performance Optimizations

This project includes several performance optimizations to improve real-time emotion detection:

### Multi-threading

- Threaded video capture for smoother video processing
- Parallel face processing for analyzing multiple faces simultaneously
- Non-blocking UI for responsive user experience

### GPU Acceleration

- TensorFlow GPU support for faster model inference
- Memory growth configuration to optimize GPU memory usage
- GPU-accelerated image processing where applicable

## Contributing

Contributions to improve the model's accuracy or add new features are welcome. Please feel free to submit a pull request.

## License

This project is licensed under the terms of the included LICENSE file.

## Acknowledgments

- Created by Hitkar Miglani
- FER2013 dataset for training and testing
- OpenCV and TensorFlow communities for their excellent documentation and tools

## Contact

For any questions or suggestions, please open an issue on GitHub or contact the project maintainer.
