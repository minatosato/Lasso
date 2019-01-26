# 
# Copyright (c) 2016-2019 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pandas as pd
import numpy as np
from lasso import Lasso

df = pd.read_csv("Boston.csv", index_col=0)
y = df.iloc[:,  13].values
df = (df - df.mean())/df.std() # 基準化
X = df.iloc[:, :13].values

model = Lasso(alpha=1.0, max_iter=1000)
model.fit(X, y)

print(model.intercept_)
print(model.coef_)

