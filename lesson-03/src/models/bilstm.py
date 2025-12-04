import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CRFLayer(nn.Module):
    """Conditional Random Field (CRF) layer manually implemented."""
    def __init__(self, num_tags, start_tag_idx, stop_tag_idx):
        super().__init__()
        self.num_tags = num_tags
        self.start_tag_idx = start_tag_idx
        self.stop_tag_idx = stop_tag_idx
        
        # Ma trận chuyển đổi (Transition Matrix)
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
        # Ngăn chặn chuyển đổi tới/từ các nhãn START/STOP không hợp lệ
        self.transitions.data[start_tag_idx, :] = -10000.
        self.transitions.data[:, stop_tag_idx] = -10000.

    def _score_sentence(self, feats, tags, mask):
        """Tính điểm của chuỗi nhãn ĐÚNG (The True Path Score)."""
        # feats: (seq_len, batch_size, num_tags)
        # tags: (seq_len, batch_size)
        
        seq_len, batch_size, _ = feats.shape
        score = torch.zeros(batch_size, device=feats.device)
        
        start_tags = torch.full((1, batch_size), self.start_tag_idx, dtype=tags.dtype, device=feats.device)
        tags_ext = torch.cat([start_tags, tags], dim=0)

        for i in range(seq_len):
            prev_tag_indices = tags_ext[i]
            curr_tag_indices = tags_ext[i+1]
            
            emission_score = feats[i, torch.arange(batch_size), curr_tag_indices]
            transition_score = self.transitions[prev_tag_indices, curr_tag_indices]
            
            score += (emission_score + transition_score) * mask[i]
        
        # Thêm Transition Score từ nhãn cuối cùng TỚI nhãn STOP
        last_tags_idx = (mask.sum(dim=0) - 1)
        last_tags = tags[last_tags_idx, torch.arange(batch_size)]
        final_transition = self.transitions[last_tags, self.stop_tag_idx]
        score += final_transition
        
        return score

    def _forward_alg(self, feats, mask):
        """Tính Log-Sum-Exp của tất cả các đường đi (Partition Function)."""
        seq_len, batch_size, num_tags = feats.shape
        
        log_alpha = torch.full((batch_size, num_tags), -10000., device=feats.device)
        log_alpha[:, self.start_tag_idx] = 0.

        for i in range(seq_len):
            log_alpha_next = torch.full((batch_size, num_tags), -10000., device=feats.device)
            
            for next_tag in range(num_tags):
                emission_score = feats[i, :, next_tag].unsqueeze(1)
                transition_scores = self.transitions[:, next_tag].unsqueeze(0) 
                
                broadcast_sum = log_alpha.unsqueeze(2) + transition_scores + emission_score
                
                log_alpha_next[:, next_tag] = torch.logsumexp(broadcast_sum, dim=1)
            
            # Cập nhật log_alpha chỉ với các bước không bị padding
            log_alpha = torch.where(mask[i].unsqueeze(1).bool(), log_alpha_next, log_alpha)

        log_prob_sum = log_alpha + self.transitions[self.stop_tag_idx].unsqueeze(0)
        final_log_sum_exp = torch.logsumexp(log_prob_sum, dim=1)
        
        return final_log_sum_exp

    def viterbi_decode(self, feats, mask):
        """Tìm chuỗi nhãn tối ưu (Viterbi Decoding)."""
        seq_len, batch_size, num_tags = feats.shape
        
        log_prob = torch.full((batch_size, num_tags), -10000., device=feats.device)
        log_prob[:, self.start_tag_idx] = 0.

        backpointers = torch.full((seq_len, batch_size, num_tags), -1, dtype=torch.long, device=feats.device)

        for i in range(seq_len):
            broadcast_score = log_prob.unsqueeze(2) + self.transitions.unsqueeze(0) 
            
            max_scores, best_tags = torch.max(broadcast_score, dim=1)
            
            new_log_prob = max_scores + feats[i]
            
            log_prob = torch.where(mask[i].unsqueeze(1).bool(), new_log_prob, log_prob)
            backpointers[i] = best_tags

        log_prob += self.transitions[:, self.stop_tag_idx]
        best_last_tag = torch.argmax(log_prob, dim=1)
        
        # Truy vết ngược (Backtracking)
        best_path = []
        curr_tag = best_last_tag
        
        # Tạo danh sách các chỉ số theo batch size để truy xuất backpointers
        batch_indices = torch.arange(batch_size, device=feats.device) 
        
        for i in range(seq_len - 1, -1, -1):
            if mask[i, 0].item(): # Chỉ truy vết ngược nếu không phải padding
                curr_tag = backpointers[i, batch_indices, curr_tag]
                best_path.insert(0, curr_tag.tolist())

        # best_path sẽ là một list các list (length = seq_len)
        return torch.tensor(best_path, device=feats.device).transpose(0, 1).tolist() # (batch_size, seq_len)

    def forward(self, feats, tags, mask):
        """Tính Loss (Log-Likelihood)"""
        score_true = self._score_sentence(feats, tags, mask)
        log_partition_func = self._forward_alg(feats, mask)
        
        return (log_partition_func - score_true).mean()



class BiLSTM(nn.Module):
    """
    Mô hình Bi-LSTM Encoder (5 lớp) + Custom CRF Layer (Decoder) cho NER.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_tags, dropout_rate=0.5):
        super().__init__()
        self.num_tags = num_tags
        
        # Giả định: START_TAG=0, STOP_TAG=1 (phải khớp với cách bạn mã hóa nhãn)
        self.start_tag_idx = 0
        self.stop_tag_idx = 1
        
        # 1. Embedding Layer
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # 2. Bi-LSTM Encoder (5 lớp)
        # Hidden size 256 (tổng 512 do Bidirectional)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim, # 256
            num_layers=num_layers, # 5 layers
            bidirectional=True,    
            dropout=dropout_rate,
            batch_first=True
        )
        
        # 3. Mapping Layer (Decoder - Ánh xạ ra Tag Space)
        lstm_output_dim = hidden_dim * 2
        self.hidden2tag = nn.Linear(lstm_output_dim, num_tags)
        
        # 4. Custom CRF Layer
        self.crf = CRFLayer(num_tags, self.start_tag_idx, self.stop_tag_idx)

    def _get_lstm_features(self, sentence):
        """Trả về Emission Scores (feats) từ Bi-LSTM."""
        # sentence: (batch_size, seq_len)
        embeds = self.dropout(self.word_embeds(sentence)) 
        
        # lstm_out: (batch_size, seq_len, 2*hidden_dim)
        lstm_out, _ = self.lstm(embeds)
        
        # tag_scores: (batch_size, seq_len, num_tags)
        tag_scores = self.hidden2tag(lstm_out) 
        
        # Chuyển về định dạng (seq_len, batch_size, num_tags) cho CRF
        return tag_scores.transpose(0, 1)

    def forward(self, sentence, tags, mask):
        """Tính Loss (Log-Likelihood Loss)"""
        feats = self._get_lstm_features(sentence)
        
        # tags và mask cần được transpose cho CRF: (seq_len, batch_size)
        tags_crf = tags.transpose(0, 1)
        mask_crf = mask.transpose(0, 1)
        
        # Loss được tính trong lớp CRF
        loss = self.crf(feats, tags_crf, mask_crf)
        return loss

    def decode(self, sentence, mask):
        """Thực hiện Viterbi Decoding (Dự đoán)"""
        feats = self._get_lstm_features(sentence)
        mask_crf = mask.transpose(0, 1)
        
        # Trả về chuỗi nhãn tối ưu: list of list (batch_size, seq_len)
        best_path = self.crf.viterbi_decode(feats, mask_crf)
        return best_path