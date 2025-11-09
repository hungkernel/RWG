# policy_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


from enum import IntEnum

class AgentAction(IntEnum):
    """Định nghĩa các hành động rời rạc mà Agent có thể chọn."""
    PLAN = 0        # Lập dàn ý
    SEARCH = 1      # Tìm kiếm (trên knowledge_db)
    WRITE = 2       # Viết một đoạn (draft) mới
    REVIEW = 3      # Đánh giá (hành động này kích hoạt Debate)
    EDIT = 4        # Chỉnh sửa (dựa trên feedback)
    WAIT = 5        # Chờ (quan trọng để tránh xung đột)

class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, K)   
        self.v  = nn.Linear(hidden, 1)   
        self.temp_head = nn.Linear(hidden, 1) 

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.body(x)
        logits = self.pi(h)        
        value  = self.v(h).squeeze(-1)  
        temp   = torch.sigmoid(self.temp_head(h)).squeeze(-1)  
        return logits, value, temp
    
    