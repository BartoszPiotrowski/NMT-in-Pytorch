import torch
import torch.nn as nn

encoder = nn.GRU(4,3, batch_first=True)
embedding = nn.Embedding(5,4)
ids = torch.tensor([[1,2,3,3,2,1], [1,2,0,0,0,0]])
print(ids)
ems = embedding(ids)
print(ems)
#print(ems.unsqueeze_(0))
print(encoder(ems))

