from typing import *
from functools import partial
from itertools import cycle

import cv2

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax

from unet import UNet


def _network(x, is_training: bool = True) -> hk.Module:
    """Semantic function to return network model"""
    return UNet()(x, is_training=is_training)


def _loss_func(alpha: float =1.e-6):
    """Semantic funciton to return loss function"""
    def l2_regularier(params):
        """Return L2 weight regularization term"""
        return 0.5*alpha*sum(
            jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params)
        )
        
    def dice_loss(params, y_train, y_pred):
        """Dice loss function"""
        onehot = jax.nn.one_hot(np.cast(y, dtype=np.float32), num_classes=2)
        prob = jax.nn.softmax(logits, axis=-1)

        inter = 2*jnp.sum(onehot, prob, axis=[1, 2])
        union = jnp.sum(onehot, axis=[1, 2]) + jnp.sum(prob, axis=[1, 2])

        loss = 1 - jnp.mean(inter/union) + l2_regularizer(params)

        return loss

    return dice_loss


def _learning_scheduler(
    init_learning_rate: float = 1.e-4,
    amplitude: float = 0.1, 
    delta: float = 1.e3,
    width: float = 1.e7
) -> float:
    """Semantic function to return scheduler of learning rate decay"""
    def schedule(count):
        return (
            self.init_learning_rate
            * (amplitude*jnp.cos(2.*np.pi*count / delta) + 1)
            * jnp.exp(-count**2. / width)
        )

    return schedule


class DataGenerator(object):                                                    
    """Data Generator for train/test batch"""                                   
    def __init__(
        self,                                                          
        sequence: list,                                                                            
        num_shuffle: int = 5, 
        mode: str ="train"
    ):                                                                          
        # Set parameters                                                        
        self.mode = mode                                  
        self.sequence = sequence
            
        # Shuffle dataset                                                     
        if mode == "train":                                                     
            self._shuffle(num_shuffle=num_shuffle)                                  
        
        # Sequence generator                                                    
        self.iter_cycle = cycle(self.sequence)                                  
                                                                                
    def _shuffle(self, num_shuffle):                                              
        [np.random.shuffle(self.sequence) for _ in range(num_shuffle)]            
         
    def __iter__(self):                                                         
        return self                                                             
                                                                                
    def __next__(self):                                                         
        if self.mode == "train":                                                
            return self.on_train_batch(next(self.iter_cycle))                                    
        elif self.mode == "test":                                               
            return self.on_test_batch(next(self.iter_cycle))                                     
                                                                                
    def on_train_batch(self, query):
        image = cv2.imread(query).astype(np.float32)

        x = image[..., 0][..., np.newaxis]/255.

        y = np.logical_and(image[..., 1] == 255, image[..., 2] == 0).astype(np.float32)[..., np.newaxis]

        return (x, y)


class TrainState(NamedTuple):
    """Training Configuration"""
    params: hk.Params
    state: hk.State
    opt_state: optax.OptState


class AuxiliaryUtility(object):
    """A class including auxiliary utilities""" 
    def save_model(self):
        """Save model weights"""
        pass

    def load_model(self):
        """Load model weights"""
        pass


class Runner(AuxiliaryUtility):
    """Model Runner"""
    def __init__(
        self,
        batch_size: int = 1,
        input_shape: tuple = (512, 512, 1),
        num_epoch: int = 10,
        step_per_epoch: int = 5000,
        init_learning_rate: float = 1.e-4,
        data_generator: Optional[DataGenerator] = None
    ):
        # Parameters
        self.batch_size = batch_size
        self.input_shape = input_shape

        self.num_epoch = num_epoch
        self.step_per_epoch = step_per_epoch
        self.init_learning_rate = init_learning_rate

        self.data_generator = data_generator

    @partial(jax.jit, static_argnums=(0,))
    def setup(self):
        """Initialize model configuration"""
        # Build network model 
        self.forward = hk.transform_with_state(_network)

        # Define optimizer
        self.learning_scheduler = _learning_scheduler(
            init_learning_rate=self.init_learning_rate,
        )
        
        self.optimizer = optax.adam(
            learning_rate=self.learning_scheduler, eps=1.e-7
        )
        
        # Define loss function
        self.loss_func = _loss_func
        
        # Initialize parameters and states
        params, state = self.forward.init(
            jax.random.PRNGKey(931016), 
            jnp.ones((self.batch_size,) + self.input_shape, dtype=np.float32), 
            is_training=True
        )
        opt_state = self.optimizer.init(params)

        # Initialize training confiuration
        train_state = valid_state = TrainState(params, state, opt_state)

        return (train_state, valid_state)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, train_state, x_train, y_train):
        """A step of training (gradient back-propagation)"""
        # Get previous configuration
        params, old_state, old_opt_state = train_state
    
        # Do prediction (update state)
        y_pred, new_state = forward.apply(params, old_state, None, x_train, is_training=True)

        # Calculate gradient of parameters 
        grads = jax.grad(self.loss_func, hax_aux=True)(params, y_train, y_pred)

        print("Loss: %f"%self.loss_func()(params, y_train, y_pred))

        # Update optimizer state
        updates, new_opt_state = self.optimizer.update(grads, old_opt_state)
        
        # Back-propagation
        new_params = optax.apply_updates(params, updates) 

        # Update training state
        train_state = valid_state = TrainState(new_params, new_state, new_opt_state)

        return train_state, valide_state

    @partial(jax.jit, static_argnums=(0,))
    def update(self, train_state, valid_state, step_size=1.e-3):
        """Update validation state"""
        # Get both configurations
        train_params, train_state, _ = train_state
        valid_params, valid_state, _ = valid_state

        # Update validation parameters
        valid_params = optax.incremental_update(
            train_params, valid_params, step_size=step_size
        )

        return TrainState(valid_params, train_state, None)

    @partial(jax.jit, static_argnums=(0,))
    def evaluate(self, params, x, y):
        return self.loss_func()(params, x, y)

    def train(self):
        # Initialize training and validation configurations
        train_state, valid_state = self.setup()

        # Run training 
        for epoch in range(self.num_epoch):
            for step in range(self.step_per_epoch):
                print("[Epoch/Step: %d/%d]"%(epoch, step))
                # Get batch data
                x_batch, y_batch = [], []
                for _ in range(self.batch_size):
                    x, y = next(data_generator)
                    x_batch.append(x)
                    y_batch.append(y)
                    x_batch = jnp.asarray(x_batch).astype(np.float32)
                    y_batch = jnp.asarray(y_batch).astype(np.float32)

                # Run a step
                #train_state = self.step(train_state, x, y)
                

import glob
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = 'platform' 

sequence = glob.glob("../data/*")

# Define data generator
data_generator = DataGenerator(
    sequence=sequence,
    num_shuffle=5,
    mode="train"
)
 

# Define model trainer
runner = Runner(
    batch_size=1,
    input_shape=(512, 512, 1),
    num_epoch=1,
    step_per_epoch=10,
    init_learning_rate=1.e-4,
    data_generator=data_generator
)

# Run training
runner.train()
    

'''
#Parallel Evaluation in Jax
#https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html

from typing import NamedTuple, Tuple
import functools

class Params(NamedTuple):
  weight: jnp.ndarray
  bias: jnp.ndarray


def init(rng) -> Params:
  """Returns the initial model params."""
  weights_key, bias_key = jax.random.split(rng)
  weight = jax.random.normal(weights_key, ())
  bias = jax.random.normal(bias_key, ())
  return Params(weight, bias)


def loss_fn(params: Params, xs: jnp.ndarray, ys: jnp.ndarray) -> jnp.ndarray:
  """Computes the least squares error of the model's predictions on x against y."""
  pred = params.weight * xs + params.bias
  return jnp.mean((pred - ys) ** 2)

LEARNING_RATE = 0.005

# So far, the code is identical to the single-device case. Here's what's new:


# Remember that the `axis_name` is just an arbitrary string label used
# to later tell `jax.lax.pmean` which axis to reduce over. Here, we call it
# 'num_devices', but could have used anything, so long as `pmean` used the same.
@functools.partial(jax.pmap, axis_name='num_devices')
def update(params: Params, xs: jnp.ndarray, ys: jnp.ndarray) -> Tuple[Params, jnp.ndarray]:
  """Performs one SGD update step on params using the given data."""

  # Compute the gradients on the given minibatch (individually on each device).
  loss, grads = jax.value_and_grad(loss_fn)(params, xs, ys)

  # Combine the gradient across all devices (by taking their mean).
  grads = jax.lax.pmean(grads, axis_name='num_devices')

  # Also combine the loss. Unnecessary for the update, but useful for logging.
  loss = jax.lax.pmean(loss, axis_name='num_devices')

  # Each device performs its own update, but since we start with the same params
  # and synchronise gradients, the params stay in sync.
  new_params = jax.tree_map(
      lambda param, g: param - g * LEARNING_RATE, params, grads)

  return new_params, loss
'''
