import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout # <--- Import LSTM

class LSTMModel(Model):
    """
    Mô hình LSTM 5 lớp cho bài toán phân loại văn bản.
    """
    def __init__(
        self, 
        vocab_size: int, 
        embedding_dim: int, 
        hidden_size: int, 
        output_size: int, 
        num_layers: int = 5, 
        dropout: float = 0.5,
        **kwargs
    ):
        super(LSTMModel, self).__init__(**kwargs)
        
        # 1. Embedding Layer
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)
        
        # 2. LSTM Layers (5 lớp)
        self.lstm_layers = []
        for i in range(num_layers):
            return_sequences = True if i < num_layers - 1 else False # Lớp cuối trả về vector đơn, các lớp trước trả về sequence
            
            # --- KHÁC BIỆT Ở ĐÂY: SỬ DỤNG LSTM ---
            lstm_layer = LSTM(
                units=hidden_size,
                return_sequences=return_sequences,
                dropout=dropout if num_layers > 1 else 0.0,
                recurrent_dropout=0.0
            )
            self.lstm_layers.append(lstm_layer)
            
        # 3. Fully Connected Layer
        self.dropout = Dropout(dropout)
        self.fc = Dense(output_size)

    def call(self, x, training=False):
        x = self.embedding(x)
        
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x, training=training)
            
        x = self.dropout(x, training=training)
        return self.fc(x)