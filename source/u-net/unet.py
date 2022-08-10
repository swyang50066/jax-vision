from typing import *

import jax 
import jax.numpy as jnp
import haiku as hk 


## Activations
"""
Common functions of Jax for neural netowrk operation, see:
https://jax.readthedocs.io/en/latest/jax.nn.html?highlight=jax.nn
"""

## Normalizations
"""
Common modules of Haiku for weight normalization, see:
https://dm-haiku.readthedocs.io/en/latest/api.html?highlight=Conv2D#normalization
"""

## Initializers
"""
The variance scaling initializer can be configured to generate other standard
initializers using the scale, mode and distribution arguments. Here are some
example configurations:
  
    ==============  ====================================================
    Name            Parameters
    ==============  ====================================================
    glorot_uniform  VarianceScaling(1.0, "fan_avg", "uniform")
    glorot_normal   VarianceScaling(1.0, "fan_avg", "truncated_normal")
    lecun_uniform   VarianceScaling(1.0, "fan_in",  "uniform")
    lecun_normal    VarianceScaling(1.0, "fan_in",  "truncated_normal")
    he_uniform      VarianceScaling(2.0, "fan_in",  "uniform")
    he_normal       VarianceScaling(2.0, "fan_in",  "truncated_normal")
    ==============  ====================================================
"""
bn_config = {
    "decay_rate": 0.99,
    "eps": 1e-3,
    "create_scale": True,
    "create_offset": True
}

InitializerType = hk.initializers.Initializer

glorot_uniform = (
    hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
)
glorot_normal = (
    hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")
)
lecun_uniform = (
    hk.initializers.VarianceScaling(1.0, "fan_in", "uniform")
)
lecun_normal = (
    hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal")
)
he_uniform = (
    hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")
)
he_normal = (
    hk.initializers.VarianceScaling(2.0, "fan_in", "truncated_normal")
)


class UpSampling2D(hk.Module):
    def __init__(self, scale: int = 2, method: Optional[str] = "nearest"):
        # Initialize subclass
        super(UpSampling2D, self).__init__()

        # Parameters
        self.scale = scale
        self.method = method
    
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        # Get input and output shape
        N, H, W, C = inputs.shape
        output_shape = (N, self.scale*H, self.scale*W, C)

        return jax.image.resize(
            inputs, shape=output_shape, method=self.method
        )


class ConvLayer2D(hk.Module):
    """2D Convolutional Layer with Jax + Haiku"""
    def __init__(
        self,
        num_feature: int,
        kernel_size: int,
        stride: int = 1,
        dilation_rate: int = 1,
        padding: str = "SAME",
        activation_func: jnp.ndarray = jax.nn.leaky_relu,
        normalization_func: Optional[hk.Module] = hk.BatchNorm,  
        kernel_initializer: Optional[InitializerType] = glorot_uniform,
        bias_initializer: Optional[InitializerType] = glorot_uniform,
        b_use_activation: bool = True,
        b_use_normalization: bool = True,
        b_use_bias: bool = True,
        name: str = "conv2d",
        **kwargs
    ):
        # Initialize subclass
        super(ConvLayer2D, self).__init__(**kwargs)

        # Parameters
        self.b_use_normalization = b_use_normalization
        self.b_use_activation = b_use_activation
    
        # Operators
        self.conv2d = hk.Conv2D(
            output_channels=num_feature,
            kernel_shape=kernel_size,
            stride=stride,
            rate=dilation_rate,
            padding=padding,
            with_bias=b_use_bias,
            w_init=kernel_initializer,
            b_init=bias_initializer,
            name=name+"_conv2d"
        )
        self.norm_func = normalization_func(name=name+"_norm", **bn_config)
        self.act_func = activation_func

    def __call__(self, inputs: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        out = self.conv2d(inputs)
        if self.b_use_normalization:
            out = self.norm_func(out, is_training)
        if self.b_use_activation:
            out = self.act_func(out)

        return out


class DeconvLayer2D(hk.Module):
    """2D Deconvolutional Layer with Jax + Haiku"""
    def __init__(
        self,
        num_feature: int,
        kernel_size: int,
        stride: int = 1,
        padding: str = "SAME",
        activation_func: jnp.ndarray = jax.nn.leaky_relu,
        normalization_func: Optional[hk.Module] = hk.BatchNorm,  
        kernel_initializer: Optional[InitializerType] = glorot_uniform,
        bias_initializer: Optional[InitializerType] = glorot_uniform,
        b_use_activation: bool = True,
        b_use_normalization: bool = True,
        b_use_bias: bool = True,
        name: str = "deconv2d",
        **kwargs
    ):
        # Initialize subclass
        super(DeconvLayer2D, self).__init__(**kwargs)

        # Parameters
        self.b_use_normalization = b_use_normalization
        self.b_use_activation = b_use_activation
    
        # Operators
        self.deconv2d = hk.Conv2DTranspose(
            output_channels=num_feature,
            kernel_shape=kernel_size,
            stride=stride,
            padding=padding,
            with_bias=b_use_bias,
            w_init=kernel_initializer,
            b_init=bias_initializer,
            name=name+"_deconv2d"
        )
        self.norm_func = normalization_func(name=name+"_norm", **bn_config)
        self.act_func = activation_func

    def __call__(self, inputs: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        out = self.deconv2d(inputs)
        if self.b_use_normalization:
            out = self.norm_func(out, is_training)
        if self.b_use_activation:
            out = self.act_func(out)

        return out


class ConvBlock2D(hk.Module):
    def __init__(
        self,
        num_repeat: int,
        num_feature: int,
        kernel_size: int,
        stride: int = 1,
        dilation_rate: int = 1,
        padding: str = "SAME",
        activation_func: jnp.ndarray = jax.nn.leaky_relu,
        normalization_func: Optional[hk.Module] = hk.BatchNorm,  
        kernel_initializer: Optional[InitializerType] = glorot_uniform,
        bias_initializer: Optional[InitializerType] = glorot_uniform,
        b_use_activation: bool = True,
        b_use_normalization: bool = True,
        b_use_bias: bool = True,
        name: str = "convblock",
        **kwargs
        ):
        # Initialize subclass 
        super(ConvBlock2D, self).__init__(**kwargs)
        
        # Operators
        self.convs = [
            ConvLayer2D(
                num_feature=num_feature,
                kernel_size=kernel_size,
                stride=stride,
                dilation_rate=dilation_rate,
                padding=padding,
                activation_func=activation_func,
                normalization_func=normalization_func,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                b_use_activation=b_use_activation,
                b_use_normalization=b_use_normalization,
                b_use_bias=b_use_bias,
                name=name+"_%d_"%index
            )
            for index in range(num_repeat)
        ]

    def __call__(self, inputs: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        for conv in self.convs:
            inputs = conv(inputs)
        
        return inputs


class UNet(hk.Module):
    """U-net with JAX and HAIKU APIs"""
    def __init__(self, **kwargs):
        # Initialize subclass
        super(UNet, self).__init__(**kwargs)

        # Parameters
        self.hk_rng = hk.PRNGSequence(931016)

        # Operators
        self.stem = ConvBlock2D(
            num_repeat=1, num_feature=64, kernel_size=3, stride=1, name="stem"
        )

        self.down01 = hk.MaxPool(
            window_shape=2, strides=2, padding="SAME", name="down01"
        )
        self.enc1 = ConvBlock2D(
            num_repeat=2, num_feature=128, kernel_size=3, stride=1, name="enc1"
        )
        
        self.down12 = hk.MaxPool(
            window_shape=2, strides=2, padding="SAME", name="down12"
        )
        self.enc2 = ConvBlock2D(
            num_repeat=2, num_feature=256, kernel_size=3, stride=1, name="enc2"
        )
        
        self.down23 = hk.MaxPool(
            window_shape=2, strides=2, padding="SAME", name="down23"
        )
        self.enc3 = ConvBlock2D(
            num_repeat=2, num_feature=512, kernel_size=3, stride=1, name="enc3"
        )
        
        self.down34 = hk.MaxPool(
            window_shape=2, strides=2, padding="SAME", name="down34"
        )
        self.enc4 = ConvBlock2D(
            num_repeat=2, num_feature=1024, kernel_size=3, stride=1, name="enc4"
        )

        self.up43 = UpSampling2D()
        self.conv3 = ConvLayer2D(
            num_feature=512, kernel_size=2, stride=1, name="conv3"
        )
        self.dec3 = ConvBlock2D(
            num_repeat=2, num_feature=512, kernel_size=3, stride=1, name="dec3"
        )

        self.up32 = UpSampling2D()
        self.conv2 = ConvLayer2D(
            num_feature=256, kernel_size=2, stride=1, name="conv2"
        )
        self.dec2 = ConvBlock2D(
            num_repeat=2, num_feature=256, kernel_size=3, stride=1, name="dec2"
        )

        self.up21 = UpSampling2D()
        self.conv1 = ConvLayer2D(
            num_feature=128, kernel_size=2, stride=1, name="conv1"
        )
        self.dec1 = ConvBlock2D(
            num_repeat=2, num_feature=128, kernel_size=3, stride=1, name="dec1"
        )

        self.up10 = UpSampling2D()
        self.conv0 = ConvLayer2D(
            num_feature=64, kernel_size=2, stride=1, name="conv0"
        )
        self.dec0 = ConvBlock2D(
            num_repeat=2, num_feature=64, kernel_size=3, stride=1, name="dec0"
        )

        self.pointwise = ConvLayer2D(
            num_feature=2, 
            kernel_size=1, 
            stride=1, 
            b_use_normalization=False, 
            b_use_activation=False, 
            b_use_bias=False,
            name="pointwise"
        )

    def __call__(self, inputs: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        # Feature embedding
        c0 = self.stem(inputs)

        # Encoding
        dc10 = self.down01(c0)
        dc11 = self.enc1(dc10)
        
        dc20 = self.down12(dc11)
        dc21 = self.enc2(dc20)
        
        dc30 = self.down23(dc21)
        dc31 = self.enc3(dc30)
        
        dc40 = self.down34(dc31)
        dc41 = self.enc4(dc40)

        # Decoding
        uc30 = self.conv3(self.up43(dc41))
        uc31 = self.dec3(jnp.concatenate([dc31, uc30], axis=-1))

        uc20 = self.conv2(self.up32(uc31))
        uc21 = self.dec2(jnp.concatenate([dc21, uc20], axis=-1))

        uc10 = self.conv1(self.up21(uc21))
        uc11 = self.dec1(jnp.concatenate([dc11, uc10], axis=-1))

        uc00 = self.conv0(self.up10(uc11))
        uc01 = self.dec0(jnp.concatenate([c0, uc00], axis=-1))

        # Representation
        fin = self.pointwise(uc01)
        if is_training:
            fin = hk.dropout(next(self.hk_rng), 0.25, fin)
        else: 
            fin = jnp.argmax(jax.nn.softmax(fin, axis=-1), axis=-1)

        return fin
