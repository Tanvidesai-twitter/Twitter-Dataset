import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from save_load import load
import joblib
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle


def classfi_report(y_test, predicted, k):

    # Classification report
    class_report = classification_report(y_test, predicted, output_dict=True)
    report_df = pd.DataFrame(class_report).transpose()

    # Plot the DataFrame
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='Blues', fmt='.2f')
    plt.title('Classification Report')
    plt.savefig(f'Results/Classification Report learning rate - {k}.png')
    plt.show()


def bar_plot(label, data1, data2, metric):

    # create data
    df = pd.DataFrame([data1, data2],
                      columns=label)
    df1 = pd.DataFrame()
    df1['Learn Rate(%)'] = [70, 80]
    df = pd.concat((df1, df), axis=1)
    # plot grouped bar chart
    df.plot(x='Learn Rate(%)',
            kind='bar',
            stacked=False)

    plt.ylabel(metric)
    plt.legend(loc='lower left')
    plt.savefig('./Results/'+metric+'.png', dpi=400)
    plt.show(block=False)


def densityplot(actual, predicted, learning_rate):

    plt.figure(figsize=(8, 6))
    sns.kdeplot(actual, color='orange', label='Actual',  fill=True)
    sns.kdeplot(predicted, color='blue', label='Predicted',  fill=True)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title("Density plot of Actual vs Predicted values")
    plt.legend()
    plt.savefig(f'Results/Density Plot Learning rate-{learning_rate}.png')
    plt.show()


def confu_plot(y_test, y_pred, lab_encoder, k):
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
    # Plot confusion matrix

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lab_encoder.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Purples, values_format='.0f', ax=ax)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plt.savefig(f'Results/Confusion Matrix Learning rate-{k}.png')
    plt.show()


def precision_recall_plot(y_test, y_pred, k):
    # Binarize the output labels
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)
    y_pred_bin = label_binarize(y_pred, classes=classes)

    # Compute Precision-Recall curve and plot
    precision = dict()
    recall = dict()
    n_classes = len(classes)

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_pred_bin[:, i])

    # Plot Precision-Recall curve for each class
    plt.figure(figsize=(10, 7))
    colors = cycle(
        ['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red', 'green', 'purple', 'brown', 'pink'])

    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='Class {0} (area = {1:0.2f})'.format(i, auc(recall[i], precision[i])))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve for multi-class data')
    plt.legend(loc="lower left")
    plt.savefig(f'Results/Precision Recall Curve - learning rate - {k}.png')
    plt.show()


def plotres():

    # learning rate -  70  and 30

    svm_70 = load('svm_70')
    naive_Bayes_70 = load('naive_Bayes_70')
    dtree_70 = load('dtree_70')
    rf_70 = load('rf_70')
    knn_70 = load('knn_70')
    proposed_70 = load('proposed_70')

    data = {
        'SVM': svm_70,
        'Naive Bayes': naive_Bayes_70,
        'DTree': dtree_70,
        'RF': rf_70,
        'KNN': knn_70,
        'PROPOSED': proposed_70
    }

    ind = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'R2 Score', 'MCC', 'Cohen’s Kappa', 'Hamming Loss', 'Jaccard Score']
    table = pd.DataFrame(data, index=ind)
    print('---------- Metrics for 70 training 30 testing ----------')
    print(table)

    table.to_excel('./Results/table_70.xlsx')

    val1 = np.array(table)

    # learning rate -  80  and 20

    svm_80 = load('svm_80')
    naive_Bayes_80 = load('naive_Bayes_80')
    dtree_80 = load('dtree_80')
    rf_80 = load('rf_80')
    knn_80 = load('knn_80')
    proposed_80 = load('proposed_80')

    data1 = {
        'SVM': svm_80,
        'Naive Bayes': naive_Bayes_80,
        'DTree': dtree_80,
        'RF': rf_80,
        'KNN': knn_80,
        'PROPOSED': proposed_80
    }

    ind =['Accuracy', 'Precision', 'Recall', 'F1 Score', 'R2 Score', 'MCC', 'Cohen’s Kappa', 'Hamming Loss', 'Jaccard Score']
    table1 = pd.DataFrame(data1, index=ind)
    print('---------- Metrics for 80 training 20 testing ----------')
    print(table1)

    val2 = np.array(table1)
    table1.to_excel('./Results/table_80.xlsx')

    metrices = [val1, val2]

    mthod = ['SVM', 'Naive Bayes', 'DTree', 'RF', 'KNN', 'Proposed']
    metrices_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'R2 Score', 'MCC', 'Cohen’s Kappa', 'Hamming Loss', 'Jaccard Score']

    # Bar plot
    for i in range(len(metrices_plot)):
        bar_plot(mthod, metrices[0][i, :], metrices[1][i, :], metrices_plot[i])

    learn_data = [70, 80]
    for k in learn_data:
        y_test = load(f'y_test_{k}')
        y_pred = load(f'predicted_{k}')
        densityplot(y_test, y_pred, k)
        classfi_report(y_test, y_pred, k)

        lab_encoder = joblib.load('Saved Data/label encoder.joblib')

        # 0 - 'A' - Agreeableness
        # 1 - 'C' - Conscientiousness
        # 2 - 'E' - Emotionality
        # 3 - 'H' - Honesty - Humility
        # 4 - 'O' - Openness to Experience
        # 5 - 'X' - Extra version

        confu_plot(y_test, y_pred, lab_encoder, k)

        precision_recall_plot(y_test, y_pred, k)

