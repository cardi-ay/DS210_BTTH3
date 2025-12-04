import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score
from tensorflow.keras.utils import Sequence # Để mô phỏng DataLoader PyTorch
from tqdm.auto import tqdm
import time

def evaluate_tf(
    model: tf.keras.Model,
    test_data, # Sử dụng tf.data.Dataset, tf.keras.utils.Sequence, hoặc generator
    n_labels: int,
    label_names: list = None # List tên các nhãn
):
    """
    Hàm đánh giá model trên tập test bằng TensorFlow/Keras.

    Args:
        model (tf.keras.Model): Model đã huấn luyện (Keras).
        test_data: Dữ liệu test (ví dụ: tf.data.Dataset hoặc Keras Sequence).
        n_labels (int): Số lượng lớp (class) của bài toán.
        label_names (list, optional): Tên của các lớp để in báo cáo chi tiết.
    """

    print("--- Bắt đầu đánh giá trên tập Test ---")

    all_preds = []
    all_labels = []

    start_time = time.time()

    # Sử dụng model.predict() để lấy tất cả dự đoán (đơn giản và hiệu quả hơn)
    # hoặc vòng lặp thủ công nếu test_data là generator/Sequence phức tạp
    
    # Cách 1: Sử dụng model.predict() (Nếu test_data là numpy array hoặc tf.data.Dataset)
    try:
        y_true = []
        # Chuyển đổi dữ liệu về numpy/list để predict
        if isinstance(test_data, tf.data.Dataset):
            # Cần lấy nhãn thật từ Dataset nếu predict không trả về
            # Giả định dataset trả về (inputs, labels)
            for inputs, labels in test_data:
                y_true.extend(labels.numpy())
            
            # Predict trả về logits/softmax
            y_pred_probs = model.predict(test_data, verbose=0)
            y_true = np.array(y_true)
            
        elif isinstance(test_data, Sequence):
            y_true = np.concatenate([test_data[i][1] for i in range(len(test_data))])
            y_pred_probs = model.predict(test_data, verbose=0)
        
        else:
            # Nếu test_data là numpy array (chỉ inputs)
            print("Lưu ý: Không tìm thấy nhãn thật. Cần data set hoặc generator có nhãn.")
            return
            
    except Exception as e:
        # Cách 2: Lặp thủ công (để xử lý các loại dữ liệu phức tạp hơn)
        print(f"Lỗi khi dùng model.predict(): {e}. Chuyển sang lặp thủ công.")
        
        y_true = []
        y_pred_probs = []
        for batch in tqdm(test_data, desc="Evaluating"):
            inputs, labels = batch # Giả sử data trả về (inputs, labels)
            
            # Forward pass
            outputs = model.predict_on_batch(inputs)
            
            # Lưu lại
            y_pred_probs.extend(outputs)
            y_true.extend(labels.numpy())
            
        y_true = np.array(y_true)
        y_pred_probs = np.array(y_pred_probs)


    # Lấy dự đoán cuối cùng (chỉ số của lớp có xác suất cao nhất)
    # y_pred_probs có shape (n_samples, n_labels)
    all_preds = np.argmax(y_pred_probs, axis=1)
    # y_true là nhãn thật (n_samples,)
    all_labels = y_true

    end_time = time.time()
    
    # --- Tính toán các chỉ số cuối cùng ---
    # Keras Loss: Để tính Test Loss, cần dùng model.evaluate() riêng.
    # Trong hàm này, ta tập trung vào các metrics.
    
    # Tính F1-Score (Macro) và Accuracy dùng Sklearn (như PyTorch đã dùng torchmetrics)
    test_f1 = f1_score(all_labels, all_preds, average="macro")
    test_acc = accuracy_score(all_labels, all_preds)

    print("\n--- 🏁 Kết quả Đánh giá trên tập Test ---")
    print(f"Thời gian đánh giá: {end_time - start_time:.2f} giây")
    # Loss: Không tính trực tiếp trong predict loop, nên bỏ qua hoặc dùng model.evaluate()
    # print(f"Test Loss: \t(Tính bằng model.evaluate() riêng biệt)")
    print(f"Test Accuracy: \t{test_acc * 100:.2f}%")
    print(f"Test F1-Score (Macro): \t{test_f1:.4f}")
    
    # --- In báo cáo chi tiết của Sklearn ---
    print("\n📊 Báo cáo chi tiết (Classification Report):")
    
    if label_names and len(label_names) == n_labels:
        report = classification_report(all_labels, all_preds, target_names=label_names)
    else:
        if label_names and len(label_names) != n_labels:
            print(f"(Lưu ý: Số lượng label_names không khớp n_labels. Sẽ dùng chỉ số 0, 1, 2...)")
        report = classification_report(all_labels, all_preds)
        
    print(report)
    
    # Trả về một dict chứa các kết quả
    return {
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "classification_report": report
    }
