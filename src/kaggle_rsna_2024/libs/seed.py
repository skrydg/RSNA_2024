import os 
import random
import numpy as np 
import tensorflow as tf 

DEFAULT_RANDOM_SEED = 2021

def seed_basic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
def seed_tf(seed=DEFAULT_RANDOM_SEED):
    tf.random.set_seed(seed)

def seed_everything(seed=DEFAULT_RANDOM_SEED):
    seed_basic(seed)
    seed_tf(seed)
