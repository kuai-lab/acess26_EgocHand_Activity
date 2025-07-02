## ===================================================
## =지우님 multi object classification branch =========

import torch
from torch import nn

class ObjClassBranch(nn.Module):
    """
    멀티라벨 객체 분류를 위한 브랜치
    """
    def __init__(self, num_obj, feature_dim):
        super(ObjClassBranch, self).__init__()
        self.num_obj = num_obj
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(self.feature_dim, self.num_obj)

    def forward(self, features):
        out = {}
        # 1. 원시 예측 점수(logits) 계산
        reg_out = self.classifier(features)
        out['reg_outs'] = reg_out

        # 2. 각 클래스의 예측 확률을 독립적으로 계산 (sigmoid 사용)
        possibilities = torch.sigmoid(reg_out)
        out['reg_possibilities'] = possibilities
        
        # 3. 확률이 0.5 이상인 모든 클래스를 예측 라벨로 선택
        pred_labels = (possibilities > 0.5).float()
        out['pred_labels'] = pred_labels
        
        return out