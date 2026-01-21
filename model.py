import torch.nn as nn

class UniversalModel(nn.Module):
    def __init__(self, num_classes=20):  # Default 20 classes, but configurable for any data
        super(UniversalModel, self).__init__()
        # Features extractor: Convolutional layers for images
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # Layer 1: Detect edges
            nn.ReLU(),  # Activation: Makes it non-linear
            nn.MaxPool2d(2, 2),  # Pool: Reduce size
            nn.Conv2d(32, 64, 3, padding=1),  # Layer 2: Deeper features
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()  # Flatten to 1D for classifier
        )
        # Classifier: Linear layer to predict classes
        self.classifier = nn.Linear(64 * 8 * 8, num_classes)  # 64 channels * 8x8 after pooling

    def forward(self, x):
        x = self.features(x)  # Extract features
        return self.classifier(x)  # Classify