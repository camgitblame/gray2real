import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1


class FaceNet(nn.Module):
    def __init__(self, pretrained="vggface2"):
        super(FaceNet, self).__init__()
        self.model = InceptionResnetV1(pretrained=pretrained).eval()

    def forward(self, x):
        """
        x: tensor of shape (B, 3, H, W) in range [0, 1]
        Returns: embeddings of shape (B, 512)
        """
        # If your inputs are in [-1, 1], rescale to [0, 1]
        if x.min() < 0:
            x = (x + 1.0) / 2.0
        return self.model(x)
