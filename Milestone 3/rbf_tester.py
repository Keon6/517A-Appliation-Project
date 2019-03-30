import np
import math
from sklearn.linear_model import LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def rbf(xtrain, ytrain, xtest, ytest):
    Y = np.asarray(ytrain)
    OLS_Estimator = LinearRegression()
    Huber_Estimator = HuberRegressor()
    Ridge_Estimator = KernelRidge(alpha=0.002, kernel="rbf", gamma=0.225)
    model2 = Ridge_Estimator.fit(xtrain, Y)
    train_fit = model2.predict(xtrain)

    Y = np.asarray(ytest)
    test_fit = model2.predict(xtest)
    rmse2 = np.sqrt(mean_squared_error(test_fit, Y))
    return(rmse2)
    #print(rmse2)





    # test_pred2 = model2.predict(xtest)
    # test_pred2[np.where(test_pred2<0)] = 0 # negative prediction data
    # test_pred2 = test_pred2.reshape(len(test_pred2),1)
    # test_pred2 = np.concatenate((IDs, test_pred2), axis=1)
    # test_pred2 = pd.DataFrame(test_pred2)
    # test_pred2.columns  = ["ID", 'Horizontal_Distance_To_Fire_Points']
    # test_pred2.to_csv(path_or_buf ="rbf_ridge.csv", index=False)