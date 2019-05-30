#!/usr/bin/env python3

import torch
import torch.nn as nn
from Encoder import Encoder


x = torch.FloatTensor([[[ 0.9059, -0.7039, -0.3376,  0.1968],[-1.0413,  0.8128,  0.0697, -0.6166],[-0.3793, -0.9851, -2.3841, -0.7003],[ 0.6076, -1.4874, -0.1079,  0.4266]]])

inputIDs = torch.randn(1, 4)
attentionMask = torch.ones_like(inputIDs)
extendedAttentionMask = attentionMask.unsqueeze(1).unsqueeze(2)
#extendedAttentionMask = extendedAttentionMask.to(dtype=next(self.parameters()).dtype)
extendedAttentionMask = (1.0 - extendedAttentionMask) * -10000.0
#tensor = torch.randn(1, 4, 4)
#print(tensor)
print()
print(extendedAttentionMask)
print()
encoder = Encoder(4, 1)
print(encoder(x, extendedAttentionMask))

