from DeepRFT_MIMO import DeepRFT
import torch

model=DeepRFT()
a=model(torch.zeros(12,3,1280,720))[0]