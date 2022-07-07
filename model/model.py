import copy
from collections import OrderedDict
from typing import Tuple, Union, Optional

import megengine as mge
import megengine.functional as F
import megengine.hub as hub
import megengine.module as M
import numpy as np

from .functional import multi_head_attention, MultiheadAttention


class CLIPConfig:
    def __init__(
        self,
        embed_dim: int,
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: Optional[int],
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int
    ):
        self.embed_dim = embed_dim
        self.image_resolution = image_resolution
        self.vision_layers = vision_layers
        self.vision_width = vision_width
        self.vision_patch_size = vision_patch_size
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.transformer_width = transformer_width
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers

    def to_dict(self):
        return copy.deepcopy(self.__dict__)


class Bottleneck(M.Module):
    """
    Bottleneck for ModifiedResNet in CLIP.
    All convolution layers have stride 1 and followed by a ReLU activation function.
    When stride > 1, An average pooling will be performed after the second convolution.
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        self.conv1 = M.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = M.BatchNorm2d(planes)
        self.relu1 = M.ReLU()

        self.conv2 = M.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = M.BatchNorm2d(planes)
        self.relu2 = M.ReLU()

        self.avgpool = M.AvgPool2d(stride) if stride > 1 else M.Identity()

        self.conv3 = M.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = M.BatchNorm2d(planes * self.expansion)
        self.relu3 = M.ReLU()

        # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
        self.downsample = (
            M.Sequential(OrderedDict([
                ("-1", M.AvgPool2d(stride)),
                ("0", M.Conv2d(inplanes, planes *
                               self.expansion, 1, stride=1, bias=False)),
                ("1", M.BatchNorm2d(planes * self.expansion))
            ]))
            if stride != 1 or inplanes != planes * Bottleneck.expansion
            else M.Identity()
        )

    def forward(self, x):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        identity = self.downsample(identity)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(M.Module):
    """
        The QKV attention layer which is used to replace average pooling.
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads: int,
        output_dim: int = None
    ):
        super().__init__()
        self.positional_embedding = mge.Parameter(
            mge.random.normal(mean=0., std=1., size=(spacial_dim ** 2 + 1, embed_dim)) / embed_dim ** 0.5)
        self.k_proj = M.Linear(embed_dim, embed_dim)
        self.q_proj = M.Linear(embed_dim, embed_dim)
        self.v_proj = M.Linear(embed_dim, embed_dim)
        self.c_proj = M.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

    def forward(self, x):
        x = F.flatten(x, 2)  # B, C, H, W -> B, C, H*W
        x = x.transpose(2, 0, 1)  # B, C, H*W -> H*W, B, C
        x = F.concat([x.mean(axis=0, keepdims=True), x], axis=0)  # H*W+1, B, C
        position_embedding = F.expand_dims(self.positional_embedding, 1)
        x_p = x + position_embedding  # H*W+1, B, C
        x, _ = multi_head_attention(
            x_p, None, None,
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            query_weight=self.q_proj.weight,
            key_weight=self.k_proj.weight,
            value_weight=self.v_proj.weight,
            attn_output_weight=self.c_proj.weight,
            query_bias=self.q_proj.bias,
            key_bias=self.k_proj.bias,
            value_bias=self.v_proj.bias,
            attn_output_bias=self.c_proj.bias,
            dropout_p=0,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(M.Module):
    """
    A modified ResNet which contains following differences：
    - The stem now includes 3 convolutions instead of 1, where max pooling is replaced by average pooling.
    - Performs anti-aliasing strided convolutions, where an average pooling is prepended to convolutions with stride > 1
    - At the final pooling layer, average pooling is replaced by a QKV attention.
    """

    def __init__(
        self,
        layers: Tuple[int, int, int, int],
        output_dim: int,
        heads: int,
        input_resolution: int = 224,
        width: int = 64
    ):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = M.Conv2d(3, width // 2, kernel_size=3,
                              stride=2, padding=1, bias=False)
        self.bn1 = M.BatchNorm2d(width // 2)
        self.relu1 = M.ReLU()
        self.conv2 = M.Conv2d(width // 2, width // 2,
                              kernel_size=3, padding=1, bias=False)
        self.bn2 = M.BatchNorm2d(width // 2)
        self.relu2 = M.ReLU()
        self.conv3 = M.Conv2d(
            width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = M.BatchNorm2d(width)
        self.relu3 = M.ReLU()
        self.avgpool = M.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(
            input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return M.Sequential(*layers)

    def stem(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class QuickGELU(M.Module):
    def __init__(self):
        super(QuickGELU, self).__init__()

    def forward(self, x):
        return x * F.sigmoid(1.702 * x)


class ResidualAttentionBlock(M.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_mask=None
    ):
        super().__init__()
        # TODO: Replace this if megengine supports MultiheadAttention one day
        self.attn = MultiheadAttention(d_model, n_head)
        self.ln_1 = M.LayerNorm(d_model)
        self.mlp = M.Sequential(OrderedDict([
            ("c_fc", M.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", M.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = M.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        if self.attn_mask is not None:
            attn_mask = mge.tensor(self.attn_mask, dtype=x.dtype, device=x.device)
        else:
            attn_mask = None
        return self.attn(x, attn_mask=attn_mask)[0]

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(M.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask=None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = M.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x):
        return self.resblocks(x)


class VisionTransformer(M.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = M.Conv2d(in_channels=3, out_channels=width,
                              kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = mge.Parameter(scale * mge.random.normal(size=(width,)))
        self.positional_embedding = mge.Parameter(
            scale * mge.random.normal(size=((input_resolution // patch_size) ** 2 + 1, width)))
        self.ln_pre = M.LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = M.LayerNorm(width)
        self.proj = mge.Parameter(scale * mge.random.normal(size=(width, output_dim)))

    def forward(self, x):
        x = self.conv1(x)  # B, width, grid, grid
        x = x.reshape(x.shape[0], x.shape[1], -1)  # B, width, grid*grid
        x = x.transpose(0, 2, 1)  # B, grid*grid, width
        # B, grid*grid + 1, width
        x = F.concat([self.class_embedding + F.zeros((x.shape[0], 1, x.shape[-1]),
                                                     dtype=x.dtype, device=x.device), x], axis=1)
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = x.transpose(1, 0, 2)  # N, L, D -> L, N, D
        x = self.transformer(x)
        x = x.transpose(1, 0, 2)  # L, N, D -> N, L, D
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(M.Module):
    def __init__(self, cfg: CLIPConfig):
        super().__init__()

        self._context_length = cfg.context_length
        self.cfg = cfg

        if isinstance(cfg.vision_layers, (tuple, list)):
            vision_heads = cfg.vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=cfg.vision_layers,
                output_dim=cfg.embed_dim,
                heads=vision_heads,
                input_resolution=cfg.image_resolution,
                width=cfg.vision_width
            )
        else:
            vision_heads = cfg.vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=cfg.image_resolution,
                patch_size=cfg.vision_patch_size,
                width=cfg.vision_width,
                layers=cfg.vision_layers,
                heads=vision_heads,
                output_dim=cfg.embed_dim
            )

        self.transformer = Transformer(
            width=cfg.transformer_width,
            layers=cfg.transformer_layers,
            heads=cfg.transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = cfg.vocab_size
        self.token_embedding = M.Embedding(cfg.vocab_size, cfg.transformer_width)
        self.positional_embedding = mge.Parameter(
            F.zeros((self._context_length, cfg.transformer_width)))
        self.ln_final = M.LayerNorm(cfg.transformer_width)

        self.text_projection = mge.Parameter(
            F.zeros((cfg.transformer_width, cfg.embed_dim)))
        self.logit_scale = mge.Parameter(F.ones([1]) * F.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        M.init.normal_(self.token_embedding.weight, std=0.02)
        M.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                M.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                M.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                M.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                M.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        M.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * \
                   ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            M.init.normal_(block.attn.in_proj.weight, std=attn_std)
            M.init.normal_(block.attn.out_proj.weight, std=proj_std)
            M.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            M.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            M.init.normal_(self.text_projection,
                           std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        mask = np.full([self._context_length, self._context_length], fill_value=float('-inf'))
        mask = np.triu(mask, k=1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    @property
    def image_resolution(self):
        return self.cfg.image_resolution

    @property
    def context_length(self):
        return self._context_length


    def model_config(self):
        model_info = ["{"]
        for k, v in self.cfg.to_dict().items():
            model_info.append(f"  {k}: {v},")
        model_info.append("}")
        print("\n".join(model_info))

    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, text):
        x = mge.tensor(self.token_embedding(text), dtype=self.dtype)
        x = x + mge.tensor(self.positional_embedding, dtype=self.dtype)
        x = x.transpose(1, 0, 2)  # N, L, D -> L, N, D
        x = self.transformer(x)
        x = x.transpose(1, 0, 2)  # L, N, D -> N, L, D
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[F.arange(x.shape[0], dtype=np.int32), F.argmax(text, axis=-1)] @ self.text_projection

        return x

    def forward(self, image, text=None):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / F.norm(image_features, axis=1, keepdims=True)
        text_features = text_features / F.norm(text_features, axis=1, keepdims=True)

        # cosine similarity as logits
        logit_scale = F.exp(self.logit_scale)
        logits_per_image = logit_scale * image_features @ text_features.transpose(1, 0)
        logits_per_text = logits_per_image.transpose(1, 0)

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def RN50(pretrained=True):
    model =  CLIP(
        embed_dim=1024,
        image_resolution=224,
        vision_layers=(3, 4, 6, 3),
        vision_width=64,
        vision_patch_size=None,
        #text
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12)
    if pretrained:
        # model.load_state_dict(
        pass
    return model


def ViT_B_32(pretrained=True):
    model =  CLIP(
        embed_dim=512,
        image_resolution=224,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=32,
        #text
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12)
    if pretrained:
        # model.load_state_dict(
        pass
    return model


def RN50x4(pretrained=True):
    model =  CLIP(
        embed_dim=640,
        image_resolution=288,
        vision_layers=(4, 6, 10, 6),
        vision_width=80,
        vision_patch_size=None,
        #text
        context_length=77,
        vocab_size=49408,
        transformer_width=640,
        transformer_heads=10,
        transformer_layers=12)
    if pretrained:
        # model.load_state_dict(
        pass
    return model

def RN50x16(pretrained=True):
    model =  CLIP(
        embed_dim=768,
        image_resolution=384,
        vision_layers=(6, 8, 18, 8),
        vision_width=96,
        vision_patch_size=None,
        #text
        context_length=77,
        vocab_size=49408,
        transformer_width=768,
        transformer_heads=12,
        transformer_layers=12)
    if pretrained:
        # model.load_state_dict(
        pass
    return model


def RN50x64(pretrained=True):
    model =  CLIP(
        embed_dim=1024,
        image_resolution=448,
        vision_layers=(3, 15, 36, 10),
        vision_width=128,
        vision_patch_size=None,
        #text
        context_length=77,
        vocab_size=49408,
        transformer_width=1024,
        transformer_heads=16,
        transformer_layers=12)
    if pretrained:
        # model.load_state_dict(
        pass
    return model

def vit_b_16(pretrained=True):
    model =  CLIP(
        embed_dim=512,
        image_resolution=224,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=16,
        #text
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12)
    if pretrained:
        # model.load_state_dict(
        pass
    return model

def vit_l_14(pretrained=True):
    model =  CLIP(
        embed_dim=768,
        image_resolution=224,
        vision_layers=24,
        vision_width=1024,
        vision_patch_size=14,
        #text
        context_length=77,
        vocab_size=49408,
        transformer_width=768,
        transformer_heads=12,
        transformer_layers=12)
    if pretrained:
        # model.load_state_dict(
        pass
    return model

def vit_l_14_336px(pretrained=True):
    model =  CLIP(
        embed_dim=768,
        image_resolution=336,
        vision_layers=24,
        vision_width=1024,
        vision_patch_size=14,
        #text
        context_length=77,
        vocab_size=49408,
        transformer_width=768,
        transformer_heads=12,
        transformer_layers=12)
    if pretrained:
        # model.load_state_dict(
        pass
    return model

def RN101(pretrained=True):
    model =  CLIP(
        embed_dim=512,
        image_resolution=224,
        vision_layers=(3, 4, 23, 3),
        vision_width=64,
        vision_patch_size=None,
        #text
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12)
    if pretrained:
        # model.load_state_dict(
        pass
    return model