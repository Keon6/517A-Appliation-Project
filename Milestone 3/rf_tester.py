import np
import math
from sklearn.linear_model import LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
def rf(xtrain, ytrain, xtest, ytest):
    Y = np.asarray(ytrain)
    OLS_Estimator = LinearRegression()
    Huber_Estimator = HuberRegressor()
    #Ridge_Estimator = KernelRidge(alpha=0.002, kernel="rbf", gamma=0.225)

    rf = RandomForestRegressor(n_estimators=400, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_depth=None, bootstrap=False)

    model2 = rf.fit(xtrain, Y)
    train_fit = model2.predict(xtrain)

    Y = np.asarray(ytest)
    test_fit = model2.predict(xtest)
    rmse2 = np.sqrt(mean_squared_error(test_fit, Y))
    return(rmse2)
    #print(rmse2)



