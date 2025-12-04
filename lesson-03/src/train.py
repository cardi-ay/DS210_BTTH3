import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
import numpy as np
import json
import os
from tqdm.keras import TqdmCallback # Tương đương tqdm.auto cho Keras

def train_keras_standard(
    model: tf.keras.Model,
    train_data, # Dữ liệu train (tf.data.Dataset, numpy arrays, hoặc Keras Sequence)
    val_data,   # Dữ liệu val
    num_epochs: int,
    model_save_path: str, # Đường dẫn lưu model (vd: "best_model.h5" hoặc thư mục)
    history_save_path: str, # Đường dẫn lưu history (vd: "history.json")
    patience: int = 10
):
    """
    Hàm train model Keras sử dụng model.fit() và Callbacks.
    Đây là cách chuẩn và hiệu quả nhất trong Keras.
    
    Args:
        model: Mô hình Keras đã được compile (có optimizer, loss, và metrics).
        train_data: Dữ liệu huấn luyện.
        val_data: Dữ liệu đánh giá.
        num_epochs: Số lượng epoch tối đa.
        model_save_path: Đường dẫn để lưu model tốt nhất.
        history_save_path: Đường dẫn để lưu lịch sử training.
        patience: Số lượng epoch chờ đợi để Early Stopping.
    """
    
    print("--- Bắt đầu training (Sử dụng Keras Standard) ---")
    print(f"Lưu model tốt nhất tại: {model_save_path}")
    print(f"Lưu lịch sử training tại: {history_save_path}")
    
    # 1. Định nghĩa Callbacks
    
    # ModelCheckpoint: Lưu model tốt nhất dựa trên Val F1-score
    # Giả định bạn đã cấu hình 'f1_score' là metric trong model.compile()
    checkpoint_cb = ModelCheckpoint(
        filepath=model_save_path,
        save_best_only=True,
        monitor='val_loss', # Thay bằng tên metric F1 của bạn
        mode='min',
        verbose=1
    )
    
    # EarlyStopping: Dừng training nếu Val F1-score không cải thiện
    early_stopping_cb = EarlyStopping(
        monitor='val_loss', # Thay bằng tên metric F1 của bạn
        mode='min',
        patience=3,
        verbose=1,
        restore_best_weights=True # Tải lại trọng số tốt nhất khi dừng
    )
    
    # Custom Callback để lưu History vào JSON sau khi training kết thúc
    class HistorySaver(Callback):
        def on_train_end(self, logs=None):
            history_data = {
                "train_loss": self.model.history.history.get('loss', []),
                "val_loss": self.model.history.history.get('val_loss', []),
                "val_f1": self.model.history.history.get('val_f1_score', []) # Thay tên metric
            }
            try:
                with open(history_save_path, 'w') as f:
                    json.dump(history_data, f, indent=4)
                print(f"\nTraining history successfully saved to {history_save_path}")
            except Exception as e:
                print(f"\nError saving history: {e}")
                
    # 2. Huấn luyện mô hình
    history = model.fit(
        train_data,
        epochs=num_epochs,
        validation_data=val_data,
        callbacks=[
            checkpoint_cb, 
            early_stopping_cb, 
            HistorySaver()
        ],
        verbose=1 # Tắt output mặc định để dùng TqdmCallback
    )

    # LƯU Ý QUAN TRỌNG:
    # Để cách tiếp cận này hoạt động, bạn phải đảm bảo mô hình được COMPILE 
    # với 'f1_score' là một trong các metrics:
    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[F1ScoreMacroKeras(name='f1_score')])
    
    return model, history.history

# ---
# KHỞI TẠO MÔ HÌNH VÀ CÁC THÀNH PHẦN (CẦN THIẾT TRƯỚC KHI GỌI HÀM)
# ---
# # Ví dụ về cách định nghĩa F1-Score custom trong Keras/TensorFlow:
# from tensorflow.keras import backend as K
# from tensorflow.keras.metrics import Metric

# class F1ScoreMacroKeras(Metric):
#     # Định nghĩa metric F1-score (cần thiết nếu muốn dùng trong model.fit)
#     # ... (code định nghĩa F1-score custom) ...
#     pass