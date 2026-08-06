"""
DeepFER — Custom Layer Definitions
====================================
Mirrors the custom layers defined in the Stage 2 training notebook (AddClsToken for ViT,
SwinBlock + PatchMerging for Swin). These must be imported (and registered) before calling
tf.keras.models.load_model() on a saved model, in case the best-performing architecture
turned out to be ViT or Swin rather than one of the standard CNN backbones.

If the best model was one of the plain CNN backbones (EfficientNet/ResNet/DenseNet/
MobileNet/ConvNeXt), this module simply isn't needed — but importing it is always safe.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="DeepFER")
class AddClsToken(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.cls = self.add_weight(name="cls_token", shape=(1, 1, self.embed_dim), initializer="zeros", trainable=True)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        cls_tokens = tf.repeat(self.cls, repeats=batch_size, axis=0)
        return tf.concat([cls_tokens, x], axis=1)


@keras.utils.register_keras_serializable(package="DeepFER")
class ExtractClsToken(layers.Layer):
    """Extracts the [CLS] token (index 0 along the sequence axis)."""
    def call(self, x):
        return x[:, 0]


def _window_partition(x, window_size):
    B = tf.shape(x)[0]
    H, W, C = x.shape[1], x.shape[2], x.shape[3]
    x = tf.reshape(x, (B, H // window_size, window_size, W // window_size, window_size, C))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    return tf.reshape(x, (-1, window_size, window_size, C))


def _window_reverse(windows, window_size, H, W):
    C = windows.shape[-1]
    B = tf.shape(windows)[0] // (H // window_size * W // window_size)
    x = tf.reshape(windows, (B, H // window_size, W // window_size, window_size, window_size, C))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    return tf.reshape(x, (B, H, W, C))


@keras.utils.register_keras_serializable(package="DeepFER")
class SwinBlock(layers.Layer):
    def __init__(self, dim, input_resolution, num_heads, window_size=3, shift_size=0, mlp_ratio=2.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.H, self.W = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size if min(self.H, self.W) > window_size else 0
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = keras.Sequential([layers.Dense(int(dim * mlp_ratio), activation="gelu"), layers.Dense(dim)])

    def call(self, x):
        H, W = self.H, self.W
        B = tf.shape(x)[0]
        shortcut = x
        x = self.norm1(x)
        x = tf.reshape(x, (B, H, W, self.dim))
        if self.shift_size > 0:
            x = tf.roll(x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))
        windows = _window_partition(x, self.window_size)
        windows = tf.reshape(windows, (-1, self.window_size * self.window_size, self.dim))
        attn_windows = self.attn(windows, windows)
        attn_windows = tf.reshape(attn_windows, (-1, self.window_size, self.window_size, self.dim))
        x = _window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = tf.roll(x, shift=(self.shift_size, self.shift_size), axis=(1, 2))
        x = tf.reshape(x, (B, H * W, self.dim))
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


@keras.utils.register_keras_serializable(package="DeepFER")
class PatchMerging(layers.Layer):
    def __init__(self, input_resolution, dim, **kwargs):
        super().__init__(**kwargs)
        self.H, self.W = input_resolution
        self.dim = dim
        self.reduction = layers.Dense(2 * dim, use_bias=False)
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        B = tf.shape(x)[0]
        x = tf.reshape(x, (B, self.H, self.W, self.dim))
        x0, x1, x2, x3 = x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :], x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]
        x = tf.concat([x0, x1, x2, x3], axis=-1)
        x = tf.reshape(x, (B, (self.H // 2) * (self.W // 2), 4 * self.dim))
        return self.reduction(self.norm(x))
