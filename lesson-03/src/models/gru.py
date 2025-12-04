import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout

class GRUModel(Model):
    """
    Mô hình GRU được xây dựng bằng TensorFlow/Keras.
    Tương đương với lớp 'gru' trong PyTorch.
    """
    def __init__(
        self, 
        vocab_size: int, 
        embedding_dim: int, 
        hidden_size: int, 
        num_layers: int, 
        num_classes: int, 
        dropout: float = 0.5,
        **kwargs # Cho phép các đối số Keras khác
    ):
        """
        Khởi tạo mô hình
        :param vocab_size: Kích thước từ điển (để tạo Embedding layer)
        :param embedding_dim: Kích thước vector embedding
        :param hidden_size: Kích thước lớp ẩn (theo yêu cầu là 256)
        :param num_layers: Số lớp GRU (theo yêu cầu là 5)
        :param num_classes: Số lượng lớp đầu ra
        :param dropout: Tỷ lệ dropout
        """
        super(GRUModel, self).__init__(**kwargs)
        
        # 1. Embedding Layer
        self.embedding = Embedding(
            input_dim=vocab_size, 
            output_dim=embedding_dim,
            mask_zero=True # Tùy chọn: giúp GRU bỏ qua padding (0)
        )
        
        # 2. Dropout Layer (Áp dụng cho Embedding, theo logic PyTorch)
        self.embed_dropout = Dropout(dropout)
        
        # 3. GRU Layer(s)
        gru_layers = []
        for i in range(num_layers):
            # return_sequences=True cho tất cả các lớp trừ lớp cuối cùng
            return_seq = True if i < num_layers - 1 else False
            
            gru_layer = GRU(
                units=hidden_size,
                return_sequences=return_seq,
                # Keras 'dropout' áp dụng cho input/output giữa các lớp RNN
                dropout=dropout if num_layers > 1 else 0.0 
            )
            gru_layers.append(gru_layer)
            
        self.gru_layers = gru_layers
        
        # 4. Dropout Layer (Áp dụng cho Hidden State cuối cùng)
        self.final_dropout = Dropout(dropout)
        
        # 5. Fully Connected Layer
        self.fc = Dense(num_classes)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Định nghĩa luồng dữ liệu đi qua mô hình (forward pass).
        
        Args:
            x (tf.Tensor): Tensor đầu vào chứa chỉ số các token.
                           Shape: (batch_size, seq_length)
                           
        Returns:
            tf.Tensor: Logits (điểm số chưa qua softmax) cho các class.
                       Shape: (batch_size, num_classes)
        """
        
        # 1. Embedding và Dropout trên Embedding
        # PyTorch: embedded = self.dropout(self.embedding(text))
        embedded = self.embedding(x)
        dropped_embedded = self.embed_dropout(embedded)
        
        # 2. Truyền qua các lớp GRU
        x_out = dropped_embedded
        for gru_layer in self.gru_layers:
            x_out = gru_layer(x_out)
            
        # x_out (từ lớp GRU cuối cùng với return_sequences=False) 
        # chính là trạng thái ẩn cuối cùng (tương đương hidden[-1] trong PyTorch)
            
        # 3. Áp dụng Dropout cho Hidden State cuối cùng
        # PyTorch: out = self.fc(self.dropout(hidden_last_layer)) 
        dropped_hidden = self.final_dropout(x_out)
        
        # 4. Qua lớp Fully Connected
        output = self.fc(dropped_hidden)
        
        return output