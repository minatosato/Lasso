# Lasso

## About
This is implementation of a coordinate descent for Lasso.

- Lasso [[Tibshirani, 1996](http://statweb.stanford.edu/%7Etibs/lasso/lasso.pdf)]
- Coordinate Descent for Lasso [J Friedman et al., [2007](http://arxiv.org/pdf/0708.1485.pdf);[2010](http://core.ac.uk/download/files/153/6287975.pdf)]

## Requirements
- NumPy
- Scikit-Learn
- Pandas (To run a sample file)

## Usage

```
git clone https://github.com/hogefugabar/Lasso
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


