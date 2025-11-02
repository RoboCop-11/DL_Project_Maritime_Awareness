"""
Maritime Domain Awareness System with YOLO, Kalman Filter, Hungarian Assignment, and LSTM Anomaly Detection

This system processes maritime SAR images to:
1. Detect and segment ships using trained YOLO model
2. Track ships using Kalman filters and Hungarian assignment
3. Detect anomalies in ship trajectories using LSTM
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import json
import os
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing functions
from preprocessing_functions.speckle_reduction import reduce_speckle_noise
from preprocessing_functions.intensity_normalization import process_intensity_normalization
from preprocessing_functions.radiometric_calibration import radiometric_calibration_pipeline

class KalmanFilter:
    """
    Kalman Filter for tracking ship positions and velocities
    State vector: [x, y, vx, vy] - position and velocity in 2D
    """
    
    def __init__(self, initial_position, dt=1.0):
        """
        Initialize Kalman filter
        
        Args:
            initial_position: [x, y] initial position
            dt: time step between measurements
        """
        self.dt = dt
        
        # State vector [x, y, vx, vy]
        self.x = np.array([initial_position[0], initial_position[1], 0.0, 0.0])
        
        # State covariance matrix
        self.P = np.eye(4) * 1000
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (we observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance
        q = 0.1  # process noise
        self.Q = np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ]) * q
        
        # Measurement noise covariance
        self.R = np.eye(2) * 10  # measurement noise
        
    def predict(self):
        """Predict next state"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2]  # return predicted position
    
    def update(self, measurement):
        """Update with measurement"""
        z = np.array(measurement)
        y = z - self.H @ self.x  # innovation
        S = self.H @ self.P @ self.H.T + self.R  # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        return self.x[:2]  # return updated position
    
    def get_state(self):
        """Get current state [x, y, vx, vy]"""
        return self.x.copy()

class ShipTracker:
    """
    Multi-object tracker for ships using Kalman filters and Hungarian assignment
    """
    
    def __init__(self, max_disappeared=10, max_distance=100):
        """
        Initialize tracker
        
        Args:
            max_disappeared: max frames a track can be missing before deletion
            max_distance: maximum distance for association
        """
        self.next_id = 0
        self.tracks = {}  # track_id -> {'kalman': KalmanFilter, 'disappeared': int, 'positions': deque}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def update(self, detections):
        """
        Update tracks with new detections
        
        Args:
            detections: list of [x, y] positions
            
        Returns:
            dict: track_id -> current position
        """
        if len(detections) == 0:
            # No detections, just predict existing tracks
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['kalman'].predict()
                self.tracks[track_id]['disappeared'] += 1
                
                # Remove tracks that have been missing too long
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.tracks[track_id]
            return {}
        
        if len(self.tracks) == 0:
            # No existing tracks, create new ones
            for detection in detections:
                self._create_track(detection)
        else:
            # Perform Hungarian assignment
            self._assign_detections(detections)
        
        # Return current track positions
        result = {}
        for track_id, track in self.tracks.items():
            if track['disappeared'] == 0:  # only active tracks
                pos = track['kalman'].get_state()[:2]
                result[track_id] = pos
                
        return result
    
    def _create_track(self, position):
        """Create new track"""
        track_id = self.next_id
        self.next_id += 1
        
        kalman = KalmanFilter(position)
        self.tracks[track_id] = {
            'kalman': kalman,
            'disappeared': 0,
            'positions': deque(maxlen=50)  # store last 50 positions
        }
        self.tracks[track_id]['positions'].append(position)
        
    def _assign_detections(self, detections):
        """Assign detections to tracks using Hungarian algorithm"""
        # Predict all tracks
        predicted_positions = []
        track_ids = []
        
        for track_id, track in self.tracks.items():
            pred_pos = track['kalman'].predict()
            predicted_positions.append(pred_pos)
            track_ids.append(track_id)
        
        if len(predicted_positions) == 0:
            # No tracks to assign to, create new ones
            for detection in detections:
                self._create_track(detection)
            return
        
        # Calculate cost matrix (distances)
        predicted_positions = np.array(predicted_positions)
        detections = np.array(detections)
        
        cost_matrix = cdist(predicted_positions, detections)
        
        # Apply distance threshold
        cost_matrix[cost_matrix > self.max_distance] = 1e6
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Track which detections and tracks were assigned
        assigned_detections = set()
        assigned_tracks = set()
        
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < 1e6:  # valid assignment
                track_id = track_ids[row]
                detection = detections[col]
                
                # Update track
                self.tracks[track_id]['kalman'].update(detection)
                self.tracks[track_id]['disappeared'] = 0
                self.tracks[track_id]['positions'].append(detection)
                
                assigned_detections.add(col)
                assigned_tracks.add(track_id)
        
        # Handle unassigned tracks
        for i, track_id in enumerate(track_ids):
            if track_id not in assigned_tracks:
                self.tracks[track_id]['disappeared'] += 1
                
                # Remove tracks that have been missing too long
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.tracks[track_id]
        
        # Create new tracks for unassigned detections
        for i, detection in enumerate(detections):
            if i not in assigned_detections:
                self._create_track(detection)
    
    def get_track_history(self, track_id, length=10):
        """Get recent position history for a track"""
        if track_id in self.tracks:
            positions = list(self.tracks[track_id]['positions'])
            return positions[-length:] if len(positions) >= length else positions
        return []

class YOLOShipDetector:
    """
    YOLO-based ship detection and segmentation
    """
    
    def __init__(self, model_path="YOLO MODELS/best.pt", confidence=0.5):
        """
        Initialize YOLO detector
        
        Args:
            model_path: path to trained YOLO model
            confidence: confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        
    def preprocess_image(self, image):
        """Apply preprocessing pipeline to image"""
        # Apply speckle noise reduction
        processed = reduce_speckle_noise(image, method='median', kernel_size=5)
        
        # Apply intensity normalization and contrast enhancement
        processed = process_intensity_normalization(
            processed,
            normalize=True,
            target_range=(0, 1),
            enhance_contrast_flag=True,
            contrast_method='clahe'
        )
        
        # Apply radiometric calibration
        processed = radiometric_calibration_pipeline(
            processed,
            scale_factor=1.1,
            offset=0,
            apply_gain=True,
            gain_value=1.2
        )
        
        return processed
    
    def detect_ships(self, image):
        """
        Detect ships in image
        
        Args:
            image: input image (BGR format)
            
        Returns:
            list: detected ship centers [(x, y), ...]
            list: bounding boxes [(x1, y1, x2, y2), ...]
            list: segmentation masks
        """
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Run YOLO inference
        results = self.model(processed_image, conf=self.confidence, verbose=False)
        
        centers = []
        boxes = []
        masks = []
        
        for result in results:
            if result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Calculate center
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    centers.append([center_x, center_y])
                    boxes.append([x1, y1, x2, y2])
                    
                    # Get segmentation mask if available
                    if result.masks is not None and i < len(result.masks):
                        mask = result.masks[i].data.cpu().numpy()
                        masks.append(mask)
                    else:
                        masks.append(None)
        
        return centers, boxes, masks

class TrajectoryAnomalyDetector:
    """
    LSTM-based anomaly detection for ship trajectories
    """
    
    def __init__(self, sequence_length=10, feature_dim=4):
        """
        Initialize anomaly detector
        
        Args:
            sequence_length: length of trajectory sequences for LSTM
            feature_dim: number of features (x, y, vx, vy)
        """
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.model = None
        self.scaler = None
        self.is_trained = False
        
        # Store trajectory data for training
        self.trajectory_data = defaultdict(list)
        
    def add_trajectory_point(self, track_id, state):
        """
        Add trajectory point for a track
        
        Args:
            track_id: unique track identifier
            state: [x, y, vx, vy] state vector
        """
        self.trajectory_data[track_id].append(state.copy())
    
    def prepare_training_data(self):
        """Prepare training data from collected trajectories"""
        sequences = []
        
        for track_id, trajectory in self.trajectory_data.items():
            if len(trajectory) >= self.sequence_length:
                # Create overlapping sequences
                for i in range(len(trajectory) - self.sequence_length + 1):
                    sequence = trajectory[i:i + self.sequence_length]
                    sequences.append(sequence)
        
        if len(sequences) == 0:
            return None, None
        
        # Convert to numpy array
        X = np.array(sequences)
        
        # Normalize data
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        
        # Reshape for scaling
        original_shape = X.shape
        X_reshaped = X.reshape(-1, self.feature_dim)
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X = X_scaled.reshape(original_shape)
        
        return X, X  # For autoencoder, input and output are the same
    
    def build_lstm_autoencoder(self):
        """Build LSTM autoencoder model"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
            
            # Encoder
            input_layer = Input(shape=(self.sequence_length, self.feature_dim))
            encoded = LSTM(32, activation='relu')(input_layer)
            
            # Decoder
            decoded = RepeatVector(self.sequence_length)(encoded)
            decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
            decoded = TimeDistributed(Dense(self.feature_dim))(decoded)
            
            # Autoencoder model
            autoencoder = Model(input_layer, decoded)
            autoencoder.compile(optimizer='adam', loss='mse')
            
            return autoencoder
            
        except ImportError:
            print("TensorFlow not available. Using simple statistical anomaly detection.")
            return None
    
    def train(self, epochs=50):
        """Train the anomaly detection model"""
        X_train, y_train = self.prepare_training_data()
        
        if X_train is None:
            print("Not enough trajectory data for training")
            return False
        
        print(f"Training with {len(X_train)} trajectory sequences")
        
        # Try to build LSTM model
        self.model = self.build_lstm_autoencoder()
        
        if self.model is not None:
            # Train LSTM autoencoder
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            # Calculate reconstruction threshold
            predictions = self.model.predict(X_train, verbose=0)
            mse = np.mean(np.power(X_train - predictions, 2), axis=(1, 2))
            self.threshold = np.percentile(mse, 95)  # 95th percentile as threshold
            
        else:
            # Fallback to statistical method
            self._train_statistical_detector(X_train)
        
        self.is_trained = True
        return True
    
    def _train_statistical_detector(self, X_train):
        """Fallback statistical anomaly detector"""
        # Calculate statistics for each feature
        self.feature_stats = {}
        
        for i in range(self.feature_dim):
            feature_data = X_train[:, :, i].flatten()
            self.feature_stats[i] = {
                'mean': np.mean(feature_data),
                'std': np.std(feature_data),
                'min': np.min(feature_data),
                'max': np.max(feature_data)
            }
        
        # Set threshold as 2 standard deviations
        self.statistical_threshold = 2.0
    
    def detect_anomaly(self, trajectory_sequence):
        """
        Detect anomaly in trajectory sequence
        
        Args:
            trajectory_sequence: list of [x, y, vx, vy] states
            
        Returns:
            float: anomaly score (higher = more anomalous)
            bool: is_anomaly flag
        """
        if not self.is_trained or len(trajectory_sequence) < self.sequence_length:
            return 0.0, False
        
        # Take last sequence_length points
        sequence = np.array(trajectory_sequence[-self.sequence_length:])
        
        if self.model is not None:
            # LSTM-based detection
            if self.scaler is not None:
                # Normalize sequence
                sequence_scaled = self.scaler.transform(sequence)
                sequence_input = sequence_scaled.reshape(1, self.sequence_length, self.feature_dim)
            else:
                sequence_input = sequence.reshape(1, self.sequence_length, self.feature_dim)
            
            # Get reconstruction
            reconstruction = self.model.predict(sequence_input, verbose=0)
            
            # Calculate reconstruction error
            mse = np.mean(np.power(sequence_input - reconstruction, 2))
            
            is_anomaly = mse > self.threshold
            return float(mse), is_anomaly
            
        else:
            # Statistical detection
            anomaly_score = 0.0
            
            for i in range(self.feature_dim):
                feature_values = sequence[:, i]
                stats = self.feature_stats[i]
                
                # Calculate z-scores
                z_scores = np.abs((feature_values - stats['mean']) / (stats['std'] + 1e-8))
                max_z_score = np.max(z_scores)
                
                anomaly_score = max(anomaly_score, max_z_score)
            
            is_anomaly = anomaly_score > self.statistical_threshold
            return float(anomaly_score), is_anomaly

class MaritimeTrackingSystem:
    """
    Complete maritime domain awareness system
    """
    
    def __init__(self, model_path="YOLO MODELS/best.pt"):
        """Initialize the complete system"""
        self.detector = YOLOShipDetector(model_path)
        self.tracker = ShipTracker()
        self.anomaly_detector = TrajectoryAnomalyDetector()
        
        # Results storage
        self.frame_results = []
        self.track_data = defaultdict(list)
        
    def process_frame(self, image, frame_id=0, timestamp=None):
        """
        Process a single frame
        
        Args:
            image: input image
            frame_id: frame identifier
            timestamp: frame timestamp
            
        Returns:
            dict: processing results
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Detect ships
        centers, boxes, masks = self.detector.detect_ships(image)
        
        # Update tracker
        tracks = self.tracker.update(centers)
        
        # Process trajectories for anomaly detection
        anomalies = {}
        for track_id, position in tracks.items():
            # Get full state from Kalman filter
            state = self.tracker.tracks[track_id]['kalman'].get_state()
            
            # Add to anomaly detector
            self.anomaly_detector.add_trajectory_point(track_id, state)
            
            # Store track data
            self.track_data[track_id].append({
                'frame_id': frame_id,
                'timestamp': timestamp,
                'position': position.tolist(),
                'state': state.tolist()
            })
            
            # Check for anomalies if we have enough data
            trajectory = [point['state'] for point in self.track_data[track_id]]
            if len(trajectory) >= 10:  # minimum sequence length
                score, is_anomaly = self.anomaly_detector.detect_anomaly(trajectory)
                anomalies[track_id] = {
                    'score': score,
                    'is_anomaly': is_anomaly
                }
        
        # Store frame results
        frame_result = {
            'frame_id': frame_id,
            'timestamp': timestamp,
            'detections': len(centers),
            'tracks': tracks,
            'anomalies': anomalies,
            'centers': centers,
            'boxes': boxes
        }
        
        self.frame_results.append(frame_result)
        
        return frame_result
    
    def train_anomaly_detector(self):
        """Train the anomaly detection model"""
        return self.anomaly_detector.train()
    
    def visualize_frame(self, image, frame_result, save_path=None):
        """
        Visualize tracking results on frame
        
        Args:
            image: original image
            frame_result: processing results from process_frame
            save_path: optional path to save visualization
        """
        vis_image = image.copy()
        
        # Draw detections and tracks
        for track_id, position in frame_result['tracks'].items():
            x, y = int(position[0]), int(position[1])
            
            # Check if this track has anomaly
            is_anomaly = False
            if track_id in frame_result['anomalies']:
                is_anomaly = frame_result['anomalies'][track_id]['is_anomaly']
            
            # Choose color based on anomaly status
            color = (0, 0, 255) if is_anomaly else (0, 255, 0)  # Red for anomaly, Green for normal
            
            # Draw track point
            cv2.circle(vis_image, (x, y), 8, color, -1)
            
            # Draw track ID
            cv2.putText(vis_image, f"ID:{track_id}", (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw trajectory history
            history = self.tracker.get_track_history(track_id, 20)
            if len(history) > 1:
                points = np.array([[int(p[0]), int(p[1])] for p in history])
                cv2.polylines(vis_image, [points], False, color, 2)
        
        # Draw bounding boxes
        for box in frame_result['boxes']:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
        # Add frame info
        info_text = f"Frame: {frame_result['frame_id']}, Ships: {frame_result['detections']}, Tracks: {len(frame_result['tracks'])}"
        cv2.putText(vis_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add anomaly count
        anomaly_count = sum(1 for a in frame_result['anomalies'].values() if a['is_anomaly'])
        if anomaly_count > 0:
            cv2.putText(vis_image, f"ANOMALIES: {anomaly_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image
    
    def export_results(self, output_dir="results"):
        """Export tracking and anomaly detection results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Export frame-by-frame results
        frame_df = pd.DataFrame(self.frame_results)
        frame_df.to_csv(f"{output_dir}/frame_results.csv", index=False)
        
        # Export track data
        all_tracks = []
        for track_id, track_points in self.track_data.items():
            for point in track_points:
                point['track_id'] = track_id
                all_tracks.append(point)
        
        track_df = pd.DataFrame(all_tracks)
        track_df.to_csv(f"{output_dir}/track_data.csv", index=False)
        
        # Export anomaly summary
        anomaly_summary = []
        for frame_result in self.frame_results:
            for track_id, anomaly_info in frame_result['anomalies'].items():
                anomaly_summary.append({
                    'frame_id': frame_result['frame_id'],
                    'track_id': track_id,
                    'anomaly_score': anomaly_info['score'],
                    'is_anomaly': anomaly_info['is_anomaly']
                })
        
        if anomaly_summary:
            anomaly_df = pd.DataFrame(anomaly_summary)
            anomaly_df.to_csv(f"{output_dir}/anomaly_results.csv", index=False)
        
        print(f"Results exported to {output_dir}/")
        
        return {
            'frame_results': frame_df,
            'track_data': track_df,
            'anomaly_results': anomaly_df if anomaly_summary else None
        }

if __name__ == "__main__":
    # Example usage
    print("Maritime Domain Awareness System initialized")
    print("Use the MaritimeTrackingSystem class to process your images")