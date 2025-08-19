import torch.nn as nn
from models.base_model import BaseModel


class DeepConvLSTM(BaseModel):
    def __init__(
        self,
        n_channels,
        n_classes,
        dataset,
        experiment="default",
        conv_kernels=64,
        kernel_size=5,
        lstm_units=128,
        lstm_layers=2,
        model="DeepConvLSTM",
    ):
        super(DeepConvLSTM, self).__init__(dataset, model, experiment)

        self.conv1 = nn.Conv2d(1, conv_kernels, (kernel_size, 1))
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))

        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(
            n_channels * conv_kernels, lstm_units, num_layers=lstm_layers
        )

        self.classifier = nn.Linear(lstm_units, n_classes)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        x = x.permute(2, 0, 3, 1)

        x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.dropout(x)
        x, h = self.lstm(x)
        x = x[-1, :, :]

        out = self.classifier(x)

        return x, out
