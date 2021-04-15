import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from glob import glob
from pprint import pprint
from utilities import get_data, get_extended_L_and_effective
from os import path
import lzma



if __name__=='__main__':
    