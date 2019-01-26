# Lasso

## Overview
This is implementation of a coordinate descent for Lasso.

- Lasso [[Tibshirani, 1996](http://statweb.stanford.edu/%7Etibs/lasso/lasso.pdf)]
- Coordinate Descent for Lasso [J Friedman et al., [2007](http://arxiv.org/pdf/0708.1485.pdf);[2010](http://core.ac.uk/download/files/153/6287975.pdf)]

See [this Japanese blog post](https://bit.ly/2MlLB22) for details of algorithm.

## Tested environment
- python==3.7.2
- numpy==1.15.4
- pandas==0.23.4 (To run a sample file)

## Usage

```
git clone https://github.com/satopirka/Lasso
cd Lasso
```

```py
from lasso import Lasso
model = Lasso(alpha=1.0, max_iter=1000).fit(X, y)
```

To run a sample program for Boston dataset,

```
python sample.py
```


