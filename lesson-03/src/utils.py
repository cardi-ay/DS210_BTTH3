import pandas as pd
import tensorflow as tf
import numpy as np
from typing import List, Tuple, Dict, Union
import os

# --- 1. Hàm load_data_from_paths ---
def load_data_from_paths(data_sets: List[Tuple[str, str, str, str]]):
    """
    Đọc dữ liệu từ file TXT.
    Input: List các tuple (tên, đường dẫn câu, đường dẫn sentiment, đường dẫn topic)
    """
    loaded_data = {}
    
    for name, sents_path, sentiments_path, topics_path in data_sets:
        try:
            if not os.path.exists(sents_path):
                print(f"❌ Không tìm thấy file: {sents_path}")
                continue

            with open(sents_path, 'r', encoding='utf-8') as f:
                sentences = [line.strip() for line in f.readlines()]
            
            with open(topics_path, 'r', encoding='utf-8') as f:
                topics = [line.strip() for line in f.readlines()]
                
            try:
                with open(sentiments_path, 'r', encoding='utf-8') as f:
                    sentiments = [line.strip() for line in f.readlines()]
            except:
                sentiments = [""] * len(sentences)

            min_len = min(len(sentences), len(topics))
            
            df = pd.DataFrame({
                'sentence': sentences[:min_len],
                'topic': topics[:min_len],
                'sentiment': sentiments[:min_len] if len(sentiments) >= min_len else [""]*min_len
            })
            
            loaded_data[name] = df
            print(f"✅ Đã load tập '{name}': {len(df)} dòng.")
            
        except Exception as e:
            print(f"❌ Lỗi khi đọc tập '{name}': {e}")
            loaded_data[name] = pd.DataFrame()

    return loaded_data


# --- 2. Class Vocab ---
class Vocab:
    def __init__(self, *args, **kwargs):
        """
        Khởi tạo Vocab - hỗ trợ nhiều format input
        """
        self.w2i: Dict[str, int] = {'<PAD>': 0, '<UNK>': 1}
        self.l2i: Dict[str, int] = {}
        self.i2l: Dict[int, str] = {}
        self.dataset = pd.DataFrame()

        # Case 1: Truyền dict
        if len(args) == 1 and isinstance(args[0], dict):
            dfs = [df for df in args[0].values() if isinstance(df, pd.DataFrame)]
            if dfs:
                self.dataset = pd.concat(dfs, ignore_index=True)
                print(f"✅ Vocab khởi tạo từ {len(dfs)} DataFrame trong dict")
        
        # Case 2: Truyền nhiều DataFrame
        elif len(args) > 1 and all(isinstance(arg, pd.DataFrame) for arg in args):
            self.dataset = pd.concat(args, ignore_index=True)
            print(f"✅ Vocab khởi tạo từ {len(args)} DataFrame")
        
        # Case 3: Truyền 1 DataFrame
        elif len(args) == 1 and isinstance(args[0], pd.DataFrame):
            self.dataset = args[0]
            print(f"✅ Vocab khởi tạo từ 1 DataFrame")
        
        # Case 4: Truyền path string
        elif len(args) == 1 and isinstance(args[0], str):
            path = args[0]
            try:
                if os.path.exists(path):
                    self.dataset = pd.read_csv(path, sep='\t', header=None, names=['sentence', 'topic'])
                    self.dataset.dropna(inplace=True)
                    print(f"✅ Vocab load từ file: {path}")
                else:
                    print(f"⚠️ Đường dẫn không tồn tại: {path}")
            except Exception as e:
                print(f"⚠️ Lỗi đọc file: {e}")

        # --- XÂY DỰNG TỪ ĐIỂN ---
        if not self.dataset.empty:
            if 'topic' in self.dataset.columns:
                labels = sorted(self.dataset['topic'].unique())
                self.l2i = {label: idx for idx, label in enumerate(labels)}
                self.i2l = {idx: label for label, idx in self.l2i.items()}

            if 'sentence' in self.dataset.columns:
                all_content = ' '.join(self.dataset['sentence'].astype(str))
                unique_words = set(all_content.split())
                
                for word in unique_words:
                    if word not in self.w2i:
                        self.w2i[word] = len(self.w2i)
        else:
            print("⚠️ Cảnh báo: Vocab khởi tạo với dữ liệu rỗng.")

    @property
    def n_labels(self) -> int:
        return len(self.l2i)
    
    @property
    def vocab_size(self) -> int:
        return len(self.w2i)
    
    def encode_sentence(self, sentence: str) -> List[int]:
        words = str(sentence).split()
        return [self.w2i.get(word, self.w2i['<UNK>']) for word in words]
    
    def encode_label(self, label: str) -> int:
        return self.l2i.get(label, -1)


# --- 3. Hàm make_tf_dataset (FIX: Sử dụng RaggedTensor + batch) ---
def make_tf_dataset(
    dataframe: pd.DataFrame, 
    vocab, 
    batch_size: int = 32, 
    is_training: bool = False,
    max_seq_len: int = None
):
    """
    ✅ Tạo dataset với padding cố định
    Đảm bảo tương thích với model có input shape cố định
    """
    if dataframe.empty:
        return tf.data.Dataset.from_tensor_slices((tf.constant([]), tf.constant([])))

    # 1. Encode dữ liệu
    X_encoded = [vocab.encode_sentence(s) for s in dataframe['sentence']]
    y_encoded = [vocab.encode_label(l) for l in dataframe['topic']]

    # 2. Xác định max_seq_len từ dữ liệu
    if max_seq_len is None:
        max_seq_len = max((len(x) for x in X_encoded), default=10)
        # Làm tròn lên bội số của 10 để không lãng phí
        max_seq_len = ((max_seq_len + 9) // 10) * 10
        print(f"  ℹ️ Tự động set max_seq_len = {max_seq_len}")
    
    # 3. Pad/truncate tất cả câu thành kích thước cố định
    X_padded = []
    for x in X_encoded:
        if len(x) < max_seq_len:
            # Pad với <PAD> token
            x = x + [vocab.w2i['<PAD>']] * (max_seq_len - len(x))
        else:
            # Truncate nếu quá dài
            x = x[:max_seq_len]
        X_padded.append(x)

    # 4. Tạo Tensor thường (KHÔNG RaggedTensor)
    X_tensor = tf.constant(X_padded, dtype=tf.int32)
    y_tensor = tf.constant(y_encoded, dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((X_tensor, y_tensor))

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(dataframe), 10000))

    # Batch (không cần padded_batch vì đã pad)
    dataset = dataset.batch(batch_size)

    if not is_training:
        dataset = dataset.cache()

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset