from projutils import load_project_data
from sklearn import linear_model
import numpy as np

trX, trY, teX, teY = load_project_data()

regr = linear_model.LinearRegression()

regr.fit(trX, trY)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
              % np.mean((regr.predict(teX) - teY) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(trX, trY))



## With ridge value
trX, trY, teX, teY = load_project_data()

regr = linear_model.Ridge(alpha=.5)

regr.fit(trX, trY)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
              % np.mean((regr.predict(teX) - teY) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(trX, trY))
