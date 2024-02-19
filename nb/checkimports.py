#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import binascii
import bokeh.plotting as bp
import collections
import copy
import Crypto
import Crypto.Random
import csv
import datetime
import datetime as dt
import email.parser
import errno
import folium
import ftfy
import functools
import getopt
import glob
import hashlib
import io
import io, os, sys, types
import itertools
import json
import keras
import keras.models as models
import logging
import math
import matplotlib.animation as manimation
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import ntpath
import numpy
import numpy as np
import numpy.linalg as la
import numpy.ma as ma
import operator
import operator 
import os
import os.path
import os, time, re, urllib
import pandas
import pandas as pd
import pickle
import PIL
import pprint
import pydot
import pylab as pl
import pylab, random
import random
import re
import requests
import scipy
import scipy.fftpack as fftpack
import seaborn as sns
import shutil
import skimage
import skimage as ski
import statsmodels.api as sm
import statsmodels.formula.api as smf
import string
import struct
import subprocess
import sympy as sym
import sys
import sys, os
import sys, time
import tempfile
import time
import time, sys
import traceback
import urllib
import uuid
import warnings
import webbrowser
import xmltodict
import zipfile

from base64 import b64encode
from Bio import Entrez, SeqIO
from Bio import SeqIO
from bokeh.models import HoverTool
from bs4 import BeautifulSoup
from collections import Counter
from collections import defaultdict
from Crypto.Hash import SHA
from Crypto.PublicKey import RSA
from Crypto.Signature import PKCS1_v1_5
from ctypes import CDLL
from difflib import SequenceMatcher
from email.parser import Parser
from functools import reduce
from IPython.core.interactiveshell import InteractiveShell
from IPython.core.pylabtools import print_figure
from IPython.display import Audio
from IPython.display import display
from IPython.display import display, clear_output
from IPython.display import display, HTML
from IPython.display import display, Image
from IPython.display import display_pretty, display_html, display_jpeg, display_png, display_json, display_latex, display_svg
from IPython.display import FileLink, FileLinks
from IPython.display import HTML
from IPython.display import HTML, Javascript, display
from IPython.display import Image
from IPython.display import Image as imaged
from IPython.display import Image, SVG, Math
from IPython.display import Javascript, display
from IPython.display import Latex
from IPython.display import Math
from IPython.display import SVG
from IPython.display import YouTubeVideo
from keras.datasets import mnist
from keras import backend as K
from keras.models import Sequential
from keras.optimizers import SGD
from math import sqrt, pi
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.patches import Polygon
from matplotlib.pyplot import specgram
from matplotlib.pyplot import violinplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import axes3d
from multiprocessing import Pool
from numpy import *
from numpy import average, linalg, dot
from numpy import log
from numpy import savetxt
from operator import mul
from PIL import Image
from plotly.graph_objs import *
from pygments.formatters import HtmlFormatter
from pygments import highlight
from pygments.lexers import PythonLexer
from pylab import *
from pylab import gray, imshow, plot, show, semilogy
from pylab import imshow, figure, zeros, plot
from random import randrange
from random import shuffle
from scipy import ndimage
from scipy import signal
from scipy import signal, ndimage
from scipy import stats
from scipy import zeros, sum
from scipy.integrate import cumtrapz
from scipy.integrate import quad, trapz
from scipy.io import wavfile
from scipy.io.wavfile import read
from scipy.ndimage import binary_closing
from scipy.ndimage import convolve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.special import jn
from scipy.stats import binom
from skimage import color
from skimage import color, filters, exposure
from skimage import filters
from skimage import morphology
from skimage.morphology import closing, square
from skimage.morphology import disk
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.sandbox.stats.multicomp import multipletests
from sympy import *
from sympy.interactive import printing
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

# cython stuff for later
#from libc.math cimport exp, sqrt, pow, log, erf
#from libc.math cimport sin

# rpy stuff
#from rpy2.robjects.packages import importr
#from rpy2.robjects.vectors import DataFrame, FloatVector

# found a python easter egg!
#from __future__ import braces

# deprecated
#from IPython.html import widgets
#from IPython.html.widgets import interact, interactive, fixed
#from IPython.html.widgets import interact_manual
#from IPython import parallel
#from IPython.kernel.zmq.datapub import publish_data
#from IPython.nbformat import current
#from IPython.parallel import Client
#from IPython.utils import inside_ipython
#from IPython.utils.path import get_ipython_package_dir
#from scipy.misc import imread
#import sklearn.cross_validation as skcross
#import sklearn.cross_validation as skcv
#from scipy.ndimage.interpolation import rotate

# perhaps deprecated?
#from keras.regularizers import ActivityRegularizer
#from keras.layers.convolutional import Convolution2D, MaxPooling2D
#from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
#from keras.layers.core import Dense, Activation, Flatten, Dropout
#from keras.layers.core import Dense, Activation, Merge
#from keras.layers.core import Dense, Dropout, Activation
#from keras.layers.core import Dense, Dropout, Activation, Flatten
#from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, AutoEncoder, Merge
#from keras.layers.core import RepeatVector, TimeDistributedDense
#from keras.layers.embeddings import Embedding
#from keras.layers.noise import GaussianNoise
#from keras.layers.recurrent import GRU, LSTM
#from keras.utils import np_utils
#from mpl_toolkits.basemap import Basemap
#from sklearn.cross_validation import cross_val_score
#from sklearn.cross_validation import ShuffleSplit
#from sklearn.cross_validation import StratifiedShuffleSplit
#from sklearn.cross_validation import StratifiedShuffleSplit  
#from sklearn.grid_search import GridSearchCV
#from bokeh.embed import notebook_div
#import keras.layers.containers as containers
#import keras.utils.visualize_util as vutil

# Not packaged
#import tumblpy
#import cPickle as pickle
#import cStringIO
#import cv2
#import kawb
#import pystan
#import scalawebsocket.WebSocket
#import skinematics as skin
#import theano
#import theano.tensor as T
#import sksound
#from itertools import combinations
#from itertools import izip
#from textblob import TextBlob
#from textblob import Word
#from urth.widgets.widget_channels import channel
#from vpython import *
#import cloudant
#import control
#import chart_studio.plotly as py
