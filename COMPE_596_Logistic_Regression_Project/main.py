# libraries used
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

'''------------------------------------------------------------------------------------------------------------------'''

# main function
def main():
    # import excel dataset as a DataFrame, then clean it
    df = pd.read_csv('diabetes.csv')
    df = df.drop(df[(df['BMI']<=0) | (df['Glucose']<=0) | (df['BloodPressure']<=0)].index)

    # split data into x_df and df_y DataFrames
    x_df = df.iloc[:,list(range(8))]
    y_df = df.iloc[:,[8]]

    # print first few rows of x_df and y_df
    print('\nPrinting first few rows of x_df (input)...\n')
    print(f'{x_df.head()}\n')
    print('Printing first few rows of y_df (output)...\n')
    print(f'{y_df.head()}\n')
    print('------------------------------------------------------------------------------------')

    # standardize/centralize x_df and convert back into DataFrame
    std = StandardScaler()
    x_df_col_labels = ['Pregnancies', 'Glucose', 'BloodPressure',
                       'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    x_df = pd.DataFrame(std.fit_transform(x_df), columns=x_df_col_labels)

    # print first few rows of standardized/centralized x_df
    print('\nPrinting first few rows of standardized x_df...\n')
    print(f'{x_df.head()}\n')
    print('------------------------------------------------------------------------------------')

    # split x_df and y_df into 70%/30% training-testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df,
                                                        train_size=0.7, test_size=0.3, random_state=100)

    # create logistic regression model using training sets
    log_reg_model = SGDClassifier(loss='log_loss', alpha=0.0001,
                                  max_iter=1000, random_state=100).fit(x_train, np.ravel(y_train))

    # create confusion matrix with training sets
    y_train_pred = log_reg_model.predict(x_train)
    train_cnf_matrix = pd.DataFrame(metrics.confusion_matrix(y_train, y_train_pred), columns=['0', '1'])

    # print confusion matrix and accuracy of training sets
    print(f'\nTraining Confusion Matrix:\n{train_cnf_matrix}\n')
    print(f'Training Accuracy = {"{:.2f}".format(log_reg_model.score(x_train, y_train) * 100)}%\n')
    print('------------------------------------------------------------------------------------')

    # create confusion matrix with testing sets
    y_test_pred = log_reg_model.predict(x_test)
    test_cnf_matrix = pd.DataFrame(metrics.confusion_matrix(y_test, y_test_pred), columns=['0', '1'])

    # print confusion matrix and accuracy of testing sets
    print(f'\nTesting Confusion Matrix:\n{test_cnf_matrix}\n')
    print(f'Testing Accuracy = {"{:.2f}".format(log_reg_model.score(x_test,y_test)*100)}%\n')

'''------------------------------------------------------------------------------------------------------------------'''

# run main function
if __name__ == "__main__":
    main()




