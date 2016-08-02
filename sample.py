#! -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import cross_validation
from lasso import Lasso

df = pd.read_csv("Boston.csv", index_col=0)
y = df.iloc[:,  13].values
df = (df - df.mean())/df.std() # 基準化
X = df.iloc[:, :13].values

model = Lasso(alpha=1.0, max_iter=1000)
model.fit(X, y)

print model.intercept_
print model.coef_

scores = cross_validation.cross_val_score(model, X, y, cv=5, scoring='mean_squared_error')
print np.mean(scores)