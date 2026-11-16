import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure local pylib is accessible
pylib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pylib")
if os.path.exists(pylib_dir) and pylib_dir not in sys.path:
    sys.path.insert(0, pylib_dir)

class CNNFeatureExtractor(nn.Module):
    """
    CNN architecture described in Section 2.3.1 and Figure 4:
    Input: (B, 2, 59, 59)
    Conv1(2, 16, 3, s=1, p=1) -> tanh -> MaxPool(2, s=2) -> (B, 16, 29, 29)
    Conv2(16, 32, 3, s=1, p=1) -> tanh -> MaxPool(2, s=2) -> (B, 32, 14, 14)
    Conv3(32, 64, 3, s=1, p=1) -> tanh -> MaxPool(2, s=2) -> (B, 64, 7, 7)
    Conv4(64, 128, 3, s=1, p=1) -> tanh -> MaxPool(2, s=2) -> (B, 128, 3, 3)
    Output: Flattened 1152 features (128 * 3 * 3 = 1152)
    """
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        
        x = torch.tanh(self.conv3(x))
        x = self.pool3(x)
        
        x = torch.tanh(self.conv4(x))
        x = self.pool4(x)
        
        x = x.view(x.size(0), -1)  # (B, 1152)
        return x


class GeoidPINN(nn.Module):
    """
    Complete PINN Architecture for Geoid Modeling (Figure 2 & Figure 4).
    Heterogeneous CNN fusion + MLP (3 hidden layers of 128 neurons each, tanh activation).
    """
    def __init__(self):
        super(GeoidPINN, self).__init__()
        # Parallel CNN feature extractors
        self.cnn_terrestrial = CNNFeatureExtractor()
        self.cnn_airborne = CNNFeatureExtractor()
        
        # Dense feature reduction: 2304 -> 16
        self.fc_dense = nn.Linear(2304, 16)
        
        # MLP: 19 -> 128 -> 128 -> 128 -> 1
        self.fc1 = nn.Linear(19, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc_out = nn.Linear(128, 1)
        
        self._init_weights()

    def _init_weights(self):
        for m in [self.fc_dense, self.fc1, self.fc2, self.fc3, self.fc_out]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_features(self, terr_grid, air_grid):
        """
        Extract 16-dim representation from gravity grids.
        """
        feat_terr = self.cnn_terrestrial(terr_grid)  # (B, 1152)
        feat_air = self.cnn_airborne(air_grid)       # (B, 1152)
        feat_cat = torch.cat([feat_terr, feat_air], dim=1)  # (B, 2304)
        feat_dense = torch.tanh(self.fc_dense(feat_cat))    # (B, 16)
        return feat_dense

    def forward_mlp(self, fused_features):
        """
        Pass 19-dim fused feature vector through MLP.
        """
        x = torch.tanh(self.fc1(fused_features))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        out = self.fc_out(x)
        return out

    def forward(self, terr_grid, air_grid, norm_coords):
        """
        Full forward pass:
        terr_grid: (B, 2, 59, 59)
        air_grid: (B, 2, 59, 59)
        norm_coords: (B, 3) -> [r_norm, nu_norm, lam_norm]
        """
        grav_feat = self.forward_features(terr_grid, air_grid)  # (B, 16)
        fused = torch.cat([grav_feat, norm_coords], dim=1)     # (B, 19)
        T_pred = self.forward_mlp(fused)                        # (B, 1)
        return T_pred


class PureDataDrivenNetwork(nn.Module):
    """
    DDN baseline (same network architecture, trained without physical constraints).
    """
    def __init__(self):
        super(PureDataDrivenNetwork, self).__init__()
        self.net = GeoidPINN()

    def forward(self, terr_grid, air_grid, norm_coords):
        return self.net(terr_grid, air_grid, norm_coords)

