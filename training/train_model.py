import os
import numpy as np
import pandas as pd
import zipfile
import json
import joblib

# TensorFlow / Keras Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE

# Plotly for High-Quality Visuals
import plotly.graph_objects as go
import plotly.express as px

# --- CONFIGURATION ---
# IMPORTANT: Adjust these paths to where your raw data is located
INPUT_DIR = "../data" 
WORKING_DIR = "./output"
EXTRACT_DIR = os.path.join(WORKING_DIR, "dataset")

# Model Hyperparameters
WINDOW_SIZE = 4000  # 0.5 seconds at 8000Hz
STRIDE = 2000       # 50% Overlap
BATCH_SIZE = 32     
EPOCHS = 80         

# Label Mapping
LEAK_MAP = {
    "Circumferential Crack": "CC", 
    "Gasket Leak": "GL",
    "Longitudinal Crack": "LC", 
    "No-leak": "NL", 
    "Orifice Leak": "OL"
}

def find_hydrophone_root(start_dir):
    """Recursively searches for the folder containing 'Branched' and 'Looped' subfolders."""
    for root, dirs, files in os.walk(start_dir):
        if "Branched" in dirs and "Looped" in dirs: 
            return root
    return None

def setup_data_source():
    """Locates the raw data or extracts it if it's in a zip file."""
    root = find_hydrophone_root(INPUT_DIR)
    if root: return root
    
    zip_path = None
    for root, dirs, files in os.walk(INPUT_DIR):
        if "Hydrophone.zip" in files:
            zip_path = os.path.join(root, "Hydrophone.zip")
            break
            
    if zip_path:
        print(f"⚡ Extracting {zip_path}...")
        os.makedirs(EXTRACT_DIR, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        return find_hydrophone_root(EXTRACT_DIR)
    return None

def read_raw_signal(filepath):
    """Reads .raw binary audio files and normalizes them."""
    try:
        signal = np.fromfile(filepath, dtype=np.int16)
        # Normalize to -1 to 1 range (16-bit audio)
        return signal.astype(np.float32) / 32768.0
    except: return None

def process_data(data_root):
    """
    Iterates through the data folders, reads signals, and applies sliding windows.
    Returns X (features) and y (labels).
    """
    print("🌊 Processing Raw Acoustic Signals...")
    X_windows = []
    y_labels = []
    
    architectures = ["Branched", "Looped"]
    for arch in architectures:
        arch_path = os.path.join(data_root, arch)
        if not os.path.exists(arch_path): continue
        
        for leak_folder in os.listdir(arch_path):
            if leak_folder not in LEAK_MAP: continue
            leak_code = LEAK_MAP[leak_folder]
            leak_path = os.path.join(arch_path, leak_folder)
            files = [f for f in os.listdir(leak_path) if f.endswith(".raw")]
            
            for filename in files: 
                signal = read_raw_signal(os.path.join(leak_path, filename))
                if signal is None or len(signal) < WINDOW_SIZE: continue
                
                # Apply Sliding Window
                for i in range(0, len(signal) - WINDOW_SIZE, STRIDE):
                    X_windows.append(signal[i : i + WINDOW_SIZE])
                    y_labels.append(leak_code)

    X = np.array(X_windows)
    # Reshape for CNN Input: (Samples, TimeSteps, Channels)
    if len(X) > 0: X = X.reshape((X.shape[0], X.shape[1], 1))
    
    print(f"✅ Generated {len(X)} samples from raw audio.")
    return X, y_labels

def build_model(input_shape, num_classes):
    """
    Constructs the Hybrid CNN-LSTM Architecture.
    
    1. CNN Layers ("The Ear"): Extract spatial features from the waveform.
    2. LSTM Layers ("The Brain"): Capture temporal dependencies.
    """
    # Functional API for better stability
    inputs = Input(shape=input_shape)
    
    # --- The Ear (CNN) ---
    x = Conv1D(32, 32, strides=2, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(4)(x)
    
    x = Conv1D(64, 16, strides=2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(4)(x)
    
    # --- The Brain (LSTM) ---
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.4)(x)
    
    # Feature Layer (Named for t-SNE extraction later)
    features = Dense(64, activation='relu', name="feature_dense")(x)
    
    # Classification Head
    outputs = Dense(num_classes, activation='softmax')(features)
    
    model = Model(inputs=inputs, outputs=outputs, name="HydroTwin_Hybrid")
    
    opt = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def save_visuals(history, model, X_test, y_test, le, y_true, y_pred):
    """Generates and saves High-Fidelity Plotly charts."""
    print("\n🎨 Generating Dashboard Visuals...")
    
    if not os.path.exists(WORKING_DIR): os.makedirs(WORKING_DIR)
    
    # 1. Accuracy Chart
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(y=history.history['accuracy'], mode='lines', name='Train'))
    fig_acc.add_trace(go.Scatter(y=history.history['val_accuracy'], mode='lines', name='Val'))
    fig_acc.update_layout(title="Model Accuracy", template="plotly_white")
    fig_acc.write_html(os.path.join(WORKING_DIR, "accuracy.html"))
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    target_names = [str(c) for c in le.classes_]
    fig_cm = px.imshow(cm, x=target_names, y=target_names, text_auto=True, color_continuous_scale='Blues')
    fig_cm.update_layout(title="Confusion Matrix")
    fig_cm.write_html(os.path.join(WORKING_DIR, "confusion_matrix.html"))

def train_pipeline():
    """Main Execution Function."""
    # 1. Data Setup
    data_root = setup_data_source()
    if not data_root: 
        print("❌ Data not found. Please ensure dataset is in '../data'.")
        return
    
    X, y = process_data(data_root)
    
    # 2. Preprocessing
    le = LabelEncoder()
    y_int = le.fit_transform(y)
    y_cat = to_categorical(y_int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y_int)
    
    # 3. Model Training
    print("\n🧠 Initializing CNN-LSTM Model...")
    model = build_model((WINDOW_SIZE, 1), len(le.classes_))
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks, verbose=1
    )
    
    # 4. Evaluation & Saving
    model.save(os.path.join(WORKING_DIR, "model.keras"))
    print("✅ Model saved to 'output/model.keras'")
    
    # Metrics
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true_indices = np.argmax(y_test, axis=1)
    
    save_visuals(history, model, X_test, y_test, le, y_true_indices, y_pred)

if __name__ == "__main__":
    train_pipeline()