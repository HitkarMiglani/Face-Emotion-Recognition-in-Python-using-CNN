import cv2
import numpy as np
import os
import time
import argparse
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json, load_model
from utils.performance import VideoStream, ParallelFaceProcessor, enable_gpu_acceleration

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def extract_features(image):
    """
    Extract features from a face image for emotion recognition.
    
    Args:
        image: The face image (grayscale, 48x48)
        
    Returns:
        Normalized feature vector
    """
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def process_face(face_img, model):
    """
    Process a face image to detect emotion.
    
    Args:
        face_img: The face image
        model: The emotion detection model
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Resize to the expected input size
        resized_face = cv2.resize(face_img, (48, 48))
        
        # Extract features
        features = extract_features(resized_face)
        
        # Get prediction directly from model
        pred = model.predict(features)
        pred_label = labels[pred.argmax()]
        confidence = np.max(pred)
        
        return {
            'emotion': pred_label,
            'confidence': confidence,
            'probabilities': pred
        }
    except Exception as e:
        print(f"Error in face processing: {e}")
        return {
            'emotion': 'error',
            'confidence': 0.0,
            'probabilities': None
        }

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Real-time Emotion Detection with Performance Optimization')
    parser.add_argument('--video', type=str, default='0', help='Path to video file or camera index (default: 0)')
    parser.add_argument('--confidence', type=float, default=0.4, help='Confidence threshold (default: 0.4)')
    parser.add_argument('--gpu', action='store_true', help='Enable GPU acceleration')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel face processing')
    parser.add_argument('--display-fps', action='store_true', help='Display FPS on the screen')
    args = parser.parse_args()
    
    # Convert video source to integer if it's a camera index
    video_source = args.video
    if video_source.isdigit():
        video_source = int(video_source)
    
    # Enable GPU acceleration if requested
    if args.gpu:
        enable_gpu_acceleration()
    
    # Initialize face detection
    haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar_file)
    
    # Load the emotion detection model
    try:
        # First try with model_from_json
        print("Loading emotion detection model...")
        json_file = open("emotiondetector.json", "r")
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model.load_weights("emotiondetector.h5")
    except Exception as e:
        print(f"Failed to load with model_from_json: {e}")
        # Alternative loading approach
        try:
            print("Attempting to load model directly from .h5 file...")
            model = load_model("emotiondetector.h5", compile=False)
        except Exception as e2:
            print(f"Failed to load model directly: {e2}")
            raise Exception("Could not load the model with any method.")
    
    print("Model loaded successfully")
    
    # Initialize parallel face processor if requested
    face_processor = None
    if args.parallel:
        face_processor = ParallelFaceProcessor(max_workers=4).start()
    
    # Initialize video stream with threading for better performance
    print(f"Starting video stream from source: {video_source}")
    video_stream = VideoStream(src=video_source).start()
    time.sleep(1.0)  # Allow the camera to warm up
    
    # Keep track of processing time
    frame_count = 0
    processing_times = []
    last_fps_print = time.time()
    fps_print_interval = 2.0  # Print FPS every 2 seconds
    
    print("Starting emotion detection. Press 'q' to quit.")
    
    # Process frames in a loop
    while not video_stream.is_stopped():
        # Get frame from the video stream
        frame = video_stream.read()
        if frame is None:
            break
        
        start_time = time.time()
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        
        # Process each face
        face_results = []
        
        if args.parallel and face_processor is not None:
            # Submit face processing tasks in parallel
            for i, (x, y, w, h) in enumerate(faces):
                face_img = gray[y:y+h, x:x+w]
                face_processor.submit_task(
                    face_img, 
                    i, 
                    lambda img: process_face(img, model)
                )
            
            # Get the results (non-blocking)
            face_results = face_processor.get_results()
            
            # Process the results
            for idx, result in face_results:
                if idx < len(faces):
                    x, y, w, h = faces[idx]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    # Display emotion label and confidence
                    emotion = result['emotion']
                    confidence = result['confidence']
                    
                    if confidence > args.confidence:
                        text = f"{emotion} ({confidence:.2f})"
                        cv2.putText(frame, text, (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, "Unknown", (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Process faces sequentially
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                result = process_face(face_img, model)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Display emotion label and confidence
                emotion = result['emotion']
                confidence = result['confidence']
                
                if confidence > args.confidence:
                    text = f"{emotion} ({confidence:.2f})"
                    cv2.putText(frame, text, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Unknown", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Calculate and display FPS
        end_time = time.time()
        processing_time = end_time - start_time
        processing_times.append(processing_time)
        
        # Keep only the last 30 frames for FPS calculation
        if len(processing_times) > 30:
            processing_times.pop(0)
        
        # Calculate FPS
        if processing_times:
            avg_processing_time = sum(processing_times) / len(processing_times)
            fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        else:
            fps = 0
        
        # Display FPS on the frame
        if args.display_fps:
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow("Emotion Detection", frame)
        
        # Print FPS to console periodically
        if time.time() - last_fps_print > fps_print_interval:
            print(f"Processing FPS: {fps:.2f}, Stream FPS: {video_stream.get_fps():.2f}")
            last_fps_print = time.time()
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        frame_count += 1
    
    # Clean up resources
    if args.parallel and face_processor is not None:
        face_processor.stop()
    
    video_stream.stop()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames")

if __name__ == "__main__":
    main()
