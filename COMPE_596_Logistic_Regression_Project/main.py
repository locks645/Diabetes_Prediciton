# libraries used
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as my_plot
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

'''------------------------------------------------------------------------------------------------------------------'''

# main function
def main():
    # import excel dataset as a DataFrame
    df = pd.read_csv('diabetes.csv')

    # split data into x_df and y_df DataFrames
    # excluding 4th col 'SkinThickness'
    x_df = df.iloc[:,[0,1,2,3,4,5,6,7]]
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
    log_reg_model = SGDClassifier(loss='log_loss',
                                  alpha=0.005, random_state=100).fit(x_train, np.ravel(y_train))

    # get training and testing set predictions with model
    y_train_pred = log_reg_model.predict(x_train)
    y_test_pred = log_reg_model.predict(x_test)
    
    # create confusion matrices with training and testing sets
    disp_train_cm = create_cm(y_train, y_train_pred, log_reg_model.classes_)
    disp_test_cm = create_cm(y_test, y_test_pred, log_reg_model.classes_)

    # create ROC plots with training and testing sets
    disp_train_roc = create_roc(y_train, y_train_pred)
    disp_test_roc = create_roc(y_test, y_test_pred)

    # print accuracy of training and testing sets
    print(f'Training Accuracy = {"{:.2f}".format(log_reg_model.score(x_train, y_train) * 100)}%\n')
    print(f'Testing Accuracy = {"{:.2f}".format(log_reg_model.score(x_test, y_test) * 100)}%\n')

    # display/show all plots
    disp_train_cm.plot()
    disp_train_roc.plot()
    disp_test_cm.plot()
    disp_test_roc.plot()
    my_plot.show()

'''------------------------------------------------------------------------------------------------------------------'''

# confusion matrix display object creation function
def create_cm(y_true, y_pred, labels):
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels)
    return metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

# ROC curve display object creation function
def create_roc(y_true, y_pred):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    return metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)

'''------------------------------------------------------------------------------------------------------------------'''

# run main function
if __name__ == "__main__":
    main()




