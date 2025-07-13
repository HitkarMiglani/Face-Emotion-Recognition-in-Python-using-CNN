import cv2
import time
import numpy as np
import threading
import queue
from typing import List, Tuple, Dict, Any, Optional

class VideoStream:
    """
    Threaded video stream class for improved performance.
    This class uses threading to parallelize video capture and processing.
    """
    def __init__(self, src=0, queue_size=128):
        """
        Initialize the video stream.
        
        Args:
            src (int or str): Video source (0 for webcam, file path for video file)
            queue_size (int): Maximum size of the queue to store frames
        """
        self.stream = cv2.VideoCapture(src)
        self.stopped = False
        self.queue = queue.Queue(maxsize=queue_size)
        self.fps_counter = FPSCounter()
        
    def start(self):
        """Start the thread to read frames from the video stream"""
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self
        
    def update(self):
        """Update method that runs in the thread"""
        self.fps_counter.start()
        while not self.stopped:
            if not self.queue.full():
                grabbed, frame = self.stream.read()
                if not grabbed:
                    self.stop()
                    return
                    
                self.queue.put(frame)
                self.fps_counter.update()
            else:
                # If queue is full, wait a bit to avoid excessive CPU usage
                time.sleep(0.01)
                
    def read(self):
        """Return the next frame from the queue"""
        return self.queue.get() if not self.queue.empty() else None
        
    def stop(self):
        """Stop the thread and release the video stream"""
        self.stopped = True
        if self.stream is not None:
            self.stream.release()
            
    def get_fps(self):
        """Get the current FPS"""
        return self.fps_counter.fps()
        
    def is_stopped(self):
        """Check if the stream is stopped"""
        return self.stopped


class FPSCounter:
    """Simple FPS counter class to measure frame processing rate"""
    def __init__(self, buffer_size=30):
        self._start = None
        self._end = None
        self._num_frames = 0
        self._buffer_size = buffer_size
        self._timestamps = []
        
    def start(self):
        """Start the timer"""
        self._start = time.time()
        return self
        
    def update(self):
        """Update the counter with a new frame"""
        self._num_frames += 1
        self._timestamps.append(time.time())
        if len(self._timestamps) > self._buffer_size:
            self._timestamps.pop(0)
        
    def fps(self):
        """Calculate the FPS"""
        if len(self._timestamps) < 2:
            return 0
            
        # Calculate FPS based on the sliding window of timestamps
        time_diff = self._timestamps[-1] - self._timestamps[0]
        frames = len(self._timestamps) - 1
        return frames / time_diff if time_diff > 0 else 0


class ParallelFaceProcessor:
    """
    Process faces in parallel using multiple threads.
    This improves performance when multiple faces are detected in a frame.
    """
    def __init__(self, max_workers=4):
        """
        Initialize the parallel face processor.
        
        Args:
            max_workers (int): Maximum number of worker threads
        """
        self.max_workers = max_workers
        self.processing_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers = []
        self.running = False
        
    def start(self):
        """Start the worker threads"""
        self.running = True
        for _ in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self.workers.append(worker)
        return self
        
    def _worker_loop(self):
        """Worker thread loop to process faces"""
        while self.running:
            try:
                task = self.processing_queue.get(timeout=0.1)
                if task is not None:
                    face_img, idx, process_func = task
                    result = process_func(face_img)
                    self.result_queue.put((idx, result))
                    self.processing_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in worker thread: {e}")
                self.processing_queue.task_done()
                
    def submit_task(self, face_img, idx, process_func):
        """
        Submit a face processing task.
        
        Args:
            face_img: The face image to process
            idx: Unique identifier for the face
            process_func: Function to process the face
        """
        self.processing_queue.put((face_img, idx, process_func))
        
    def get_results(self, timeout=0.1):
        """
        Get all available results.
        
        Returns:
            List of (idx, result) tuples
        """
        results = []
        try:
            while True:
                result = self.result_queue.get(block=False)
                results.append(result)
                self.result_queue.task_done()
        except queue.Empty:
            pass
        return results
        
    def stop(self):
        """Stop all worker threads"""
        self.running = False
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=1.0)
        self.workers = []


# GPU optimization functions
def enable_gpu_acceleration():
    """
    Configure TensorFlow to use GPU acceleration if available.
    Returns whether GPU was successfully enabled.
    """
    try:
        import tensorflow as tf
        
        # Check if GPU is available
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            try:
                # Enable memory growth to avoid allocating all GPU memory at once
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Set TensorFlow to use the first GPU
                tf.config.set_visible_devices(gpus[0], 'GPU')
                
                # Print GPU information
                print(f"GPU acceleration enabled: {len(gpus)} GPU(s) available")
                return True
                
            except RuntimeError as e:
                print(f"Error enabling GPU acceleration: {e}")
                return False
        else:
            print("No GPU available, using CPU.")
            return False
            
    except ImportError:
        print("TensorFlow not installed or cannot be imported.")
        return False
