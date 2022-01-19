# Implementação e treinamento da rede
from torch import nn


class VGG16(nn.Module):

    def __init__(self, in_size, out_size):
        super(VGG16, self).__init__()

        self.features = nn.Sequential(
            # ConvBlock 1
            nn.Conv2d(in_channels=in_size, out_channels=64, kernel_size=3, stride=1, padding=1), # entrada (b, in_size, 224, 224) saida (b, 64, 224, 224)
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # entrada (b, 64, 224, 224) saida (b, 64, 224, 224)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # entrada (b, 64, 224, 224) saida (b, 64, 112, 112)

            # ConvBlock 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), # entrada (b, 64, 112, 112) saida (b, 128, 112, 112)
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # entrada (b, 128, 112, 112) saida (b, 128, 112, 112)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # entrada (b, 128, 112, 112) saida (b, 128, 56, 56)

            #ConvBlock 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1), # entrada (b, 128, 56, 56) saida (b, 256, 56, 56)
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), # entrada (b, 256, 56, 56) saida (b, 256, 56, 56)
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), # entrada (b, 256, 56, 56) saida (b, 256, 56, 56)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # entrada (b, 256, 56, 56) saida (b, 256, 28, 28)

            #ConvBlock 4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1), # entrada (b, 256, 28, 28) saida (b, 512, 28, 28)
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # entrada (b, 512, 28, 28) saida (b, 512, 28, 28)
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # entrada (b, 512, 28, 28) saida (b, 512, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # entrada (b, 512, 28, 28) saida (b, 512, 14, 14)

            #ConvBlock 5
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # entrada (b, 512, 14, 14) saida (b, 512, 14, 14)
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # entrada (b, 512, 14, 14) saida (b, 512, 14, 14)
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # entrada (b, 512, 14, 14) saida (b, 512, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # entrada (b, 512, 14, 14) saida (b, 512, 7, 7)
            nn.Flatten(),   # entrada (b, 512, 7, 7) saida (b, 512 * 7 * 7) = (b, 25088)
        )

        self.out = nn.Sequential(
            # DenseBlock
            nn.Linear(25088, 4096), # entrada (b, 25088) saida (b, 4096)
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), # entrada (b, 4096) saida (b, 4096)
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, out_size), # entrada (b, 4096) saida (b, out_size)
            nn.ReLU(),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, X):
        feature = self.features(X)
        output = self.out(feature)
        return output
