import data_preprocessor as dp
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

def main():
    data = dp.preprocess_data()
    print("Data loaded and preprocessed successfully.")
    print(data.head())

    X = data.drop(columns=['target'])
    X = sm.add_constant(X)
    y = data['target']

    model = sm.OLS(y, X)
    result = model.fit()
    print(result.summary())

    plt.figure(figsize=(45,18))
    sns.lineplot(y)
    sns.lineplot(result.predict(X))
    plt.show()


if __name__ == '__main__':
    main()