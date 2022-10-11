import torch.nn as nn

class Linear_Unit(nn.Module):
    def __init__(self, dimension_in,dimension_out,add_batch_norm=True):
        super(Linear_Unit, self).__init__()

        self.fc = nn.Linear(dimension_in, dimension_out)
        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(dimension_out)
        
    def forward(self,x):
        
        x = self.fc(x)
        
        if self.add_batch_norm:
            x = self.batch_norm(x) 

        return x
