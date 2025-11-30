import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
import statsmodels.api as sm




def fcpu():


    cols = ["Vendor", "Model", "MYCT", "MMIN",
            "MMAX", "CACH", "CHMIN", "CHMAX",
            "PRP", "ERP"]

    df = pd.read_csv("machine.data", names=cols)
    x = df.iloc[:,2:-2].values
    y = df.iloc[:,8].values
    return x, y

### optimize the model with backwards elimination to select that most significant predictors

def backwards_elimination(x, y, sL = 0.05):


    x_opt = x.copy()
    
    num_var = x.shape[1]
    for _ in range(num_var):
        ols_model = sm.OLS(y, sm.add_constant(x_opt)).fit()
        max_pvalue = max(ols_model.pvalues[1:])
        if max_pvalue > sL:
            max_p_index = np.argmax(ols_model.pvalues[1:])
            x_opt = np.delete(x_opt, max_p_index, axis=1)
        else:
            break


    return x_opt, ols_model

def training_prep():
    x, y = fcpu()

    x_train, x_test, y_train, y_test = train_test_split(x ,y,test_size =  0.2, random_state = 0)
    return x_train, x_test, y_train, y_test

def standardizing():
    x_train, x_test, y_train, y_test = training_prep()


    sc_x  = StandardScaler()
    x_train_scaled = sc_x.fit_transform(x_train)
    x_test_scaled = sc_x.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test

def scaled_x():
    x_train, x_test, y_train, y_test = standardizing()

    # backwards elimination implementation
    x_full = np.vstack((x_train, x_test))
    y_full = np.concatenate((y_train, y_test))

    x_full_const = np.append(arr=np.ones((x_full.shape[0], 1)), values=x_full, axis=1)
    x_reduced, model = backwards_elimination(x_full_const, y_full, sL = 0.05)

    # reduced matrix into train/tet
    n_train = x_train.shape[0]

    x_reduced_train = x_reduced[:n_train, :]
    x_reduced_test = x_reduced[n_train:, :]

    return x_reduced_train, x_reduced_test, y_train, y_test

def regression_model():
    x_reduced_train, x_reduced_test, y_train, y_test = scaled_x()


    regressor = LinearRegression()
    regressor.fit(x_reduced_train, y_train)

    y_pred = regressor.predict(x_reduced_test)

    return y_pred, y_test

def data_comparision():
    y_pred, y_test = regression_model()

    df1 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
    df1.plot(kind='bar')
    plt.xlabel("Sample")
    plt.ylabel("Published Relative perforamnce")
    plt.title("Supervised Regression Model of CPU Performance")
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()






























def test_preprocessing():
    print("Running preprocessing test...\n")

    x_train, x_test, y_train, y_test = standardizing()

    test_pass = True

    print("Train shape:", x_train.shape)
    print("Test shape:", x_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    if x_train.shape[1] != x_test.shape[1]:
        print("ERROR: Feature dimension mismatch between train/set.")
        test_pass = False
    
    # NaN checks
    if np.isnan(x_train).any() or np.isnan(x_test).any():
        print("ERROR: NaNs found in scaled data")
        test_pass = False

    train_mean = np.mean(x_train, axis=0)
    train_std = np.std(x_train, axis=0)

    print("\nTrain feature means (should be ~0):", train_mean)
    print("Train feature std (should be ~1): ", train_std)

    if not np.allclose(train_mean, 0, atol=1e-6):
        print("Warning: Means are not close to zero.")
    if not np.allclose(train_std, 1, atol=1e-6):
        print("Warning: Stds are not close to one")
    

    if len(y_train) == 0 or len(y_test) == 0:
        print("ERROR: y_train or y_test is empty")
        test_pass = False

    print("\nPreprocessing test complete")
    if test_pass:
        print("Preprocessing PASSED")
    else:
        print("Preprocessing FAILED")



if __name__ == "__main__":
    data_comparision()








