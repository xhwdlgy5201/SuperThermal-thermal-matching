from torch import nn


class Enhance(nn.Module):
    def __init__(self):
        super(ten, self).__init__()
        self.model = nn.Sequential(

            nn.Conv2d(1,64,7,stride=1,padding=3),
            nn.ReLU(True),

            nn.Conv2d(64,32,5,stride=1,padding=2),
            nn.ReLU(True),

            nn.Conv2d(32,32,3,stride=1,padding=1),
            nn.ReLU(True),

            nn.Conv2d(32,1,3,stride=1,padding=1)
        )

    def forward(self,x):
        x = self.model(x)
        return x