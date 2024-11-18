import numpy as np
from dtw import *

import numpy as np
from dtw import *
x=[2., 0., 1., 1., 2., 4., 2., 1., 2., 0.]
y=[1., 1., 2., 4., 2., 1., 2., 0.]

ds = dtw(y, x,keep_internals=True, step_pattern=asymmetric)
ds.plot(type="twoway",offset=-2)
