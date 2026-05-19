import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositionClassifier(nn.Module):

    def __init__(self, input_dim, num_classes, normalization_sign=False):
        super().__init__()
        half_input_dim = int(input_dim / 2)

        self.mlp = nn.Linear(input_dim, half_input_dim)
        self.fc = nn.Linear(half_input_dim, num_classes)
        self.normalization = normalization_sign

    def forward(self, f1, f2):
        """
        f1: other modality (e.g. audio or vision)
        f2: video modality
        Returns: (logits, fused feature). The fused feature is f1 plus a
        residual learned from concat([f1, f2]).
        """
        if self.normalization:
            f1_n = F.normalize(f1, dim=1)
            f2_n = F.normalize(f2, dim=1)
            residual = torch.cat((f1_n, f2_n), 1)
        else:
            residual = torch.cat((f1, f2), 1)

        residual = self.mlp(residual)
        feature = f1 + residual

        out = self.fc(feature)
        return out, feature


if __name__ == "__main__":
    from torchsummary import summary

    model = CompositionClassifier(input_dim=1024, num_classes=2, normalization_sign=True)
    summary(model, input_size=[(283, 512), (283, 512)], batch_size=12, device="cpu")
