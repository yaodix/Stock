import numpy as np
from dtw import *
import matplotlib.pyplot as plt
import numpy as np
from dtw import *
x=[2., 0., 1., 1., 2., 4., 2., 1., 2., 0.]
y=        [1., 1., 2., 4., 2., 1., 2., 0.]
from sktime.datasets import load_osuleaf

# load an example time series panel in numpy mtype
X, _ = load_osuleaf(return_type="pd-multiindex")

X1 = X.loc[0]  # leaf 0
X2 = X.loc[1]  # leaf 1
from sktime.utils.plotting import plot_series

plot_series(X1, X2, labels=["leaf_1", "leaf_2"])

from sktime.alignment.dtw_python import AlignerDTW

# use dtw-python package for aligning
# simple univariate alignment algorithm with default params
aligner = AlignerDTW()
aligner.fit([X1, X2])  # series to align need to be passed as list
# obtain the aligned versions of the two series
X1_al, X2_al = aligner.get_aligned()

from sktime.utils.plotting import plot_series

plot_series(
    X1_al.reset_index(drop=True),
    X2_al.reset_index(drop=True),
    labels=["leaf_1", "leaf_2"],
)
# this is the distance between the two time series we aligned
print(aligner.get_distance())