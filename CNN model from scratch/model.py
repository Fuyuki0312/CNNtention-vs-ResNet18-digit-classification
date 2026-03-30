from torch import nn
class Model_detecting_number(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=(3, 3),
                      padding=1,
                      stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=(3, 3),
                      padding=1,
                      stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=(3, 3),
                      padding=1,
                      stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=(3, 3),
                      padding=1,
                      stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((22, 35)),
            nn.Flatten(),
            nn.Linear(in_features=32*22*35, out_features=10)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        return x
