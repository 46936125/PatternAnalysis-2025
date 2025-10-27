import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class LayerNorm2d(nn.Module):
    """LayerNorm for channels-first data (N, C, H, W)."""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class DropPath(nn.Module):
    """Stochastic depth regularization."""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block: depthwise conv + MLP + residual."""
    def __init__(self, dim, drop_path=0.4, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(x)
        return x


class ConvNeXtTiny(nn.Module):
    """ConvNeXt-Tiny built fully from scratch."""
    def __init__(self, num_classes=2, in_chans=3, dropout_rate=0.4):
        super().__init__()

        depths = [3, 3, 9, 3]
        dims = [96, 192, 384, 768]
        self.dropout_rate = dropout_rate

        # Downsampling layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0])
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            down = nn.Sequential(
                LayerNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(down)

        # Stages of ConvNeXt blocks
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, self.dropout_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[
                ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j])
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

        # Final layers
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = x.mean([-2, -1])
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.dropout(x)
        x = self.head(x)
        return x


class ADNIConvNext(nn.Module):
    """
    Custom ConvNeXt-Tiny for ADNI.
    """
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super().__init__()
        self.backbone = ConvNeXtTiny(num_classes=num_classes, dropout_rate=dropout_rate)

        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)

    def update_dropout_rate(self, dropout_rate):
        self.backbone.dropout = nn.Dropout(dropout_rate)
        self.backbone.dropout_rate = dropout_rate


class ModelEMA:
    """Exponential Moving Average for model weights."""
    def __init__(self, model, decay=0.999):
        # Make a copy of the model for EMA
        self.ema_model = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            ema_params = dict(self.ema_model.named_parameters())
            for n, p in model.named_parameters():
                if n in ema_params:
                    ema_params[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        """Load EMA weights into the main model temporarily."""
        model.load_state_dict(self.ema_model.state_dict())
