from projutils import load_project_data
from sklearn import svm
import numpy as np

trX, trY, teX, teY = load_project_data()

# Linear Model
model = svm.SVR(kernel='linear')
model.fit(trX, trY)

# The coefficients
print('Coefficients: \n', model.coef_)
# The mean squared error
print("Mean squared error: %.2f"
              % np.mean((model.predict(teX) - teY) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % model.score(trX, trY))


# RBG Kernel
model = svm.SVR(kernel='rbf')
model.fit(trX, trY)

# The mean squared error
print("Mean squared error: %.2f"
              % np.mean((model.predict(teX) - teY) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % model.score(trX, trY))


# Polynomial Kernel
model = svm.SVR(kernel='poly')
model.fit(trX, trY)

# The mean squared error
print("Mean squared error: %.2f"
              % np.mean((model.predict(teX) - teY) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % model.score(trX, trY))
