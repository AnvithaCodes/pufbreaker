"""
PUFBreaker: ML Security Analysis for Physical Unclonable Functions

A Python library for simulating PUFs and testing machine learning attacks.
"""

__version__ = "0.1.0"

from .arbiter_puf import ArbiterPUF
from .xor_puf import XORPUF
from .lr_attack import LRAttack
from .mlp_attack import MLPAttack
from . import utils

__all__ = [
    'ArbiterPUF',
    'XORPUF', 
    'LRAttack',
    'MLPAttack',
    'utils'
]