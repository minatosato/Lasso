#! -*- coding: utf-8 -*-

import pandas as pd
from lasso import Lasso

df = pd.read_csv("Boston.csv", index_col=0)
y = df.iloc[:,  13].values
df = (df - df.mean())/df.std() # 基準化
X = df.iloc[:, :13].values

model = Lasso(alpha=1.0, max_iter=1000).fit(X=X, y=y)

print model.intercept_
print model.coef_