from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def calculate_metrics(predicted, scores, y_true, num_classes):
        accuracy = accuracy_score(y_true, predicted)
        f1 = f1_score(y_true, predicted, average='macro')
        precision = precision_score(y_true, predicted, average='macro', zero_division=1)
        recall = recall_score(y_true, predicted, average='macro')
        mcc = matthews_corrcoef(y_true, predicted)
        auc_roc = roc_auc_score(y_true, scores, average='macro', multi_class='ovr')

        try:
            auc_roc = roc_auc_score(y_true, scores, average='macro', multi_class='ovr')
        except ValueError as e:
            if "Number of classes in y_true not equal to the number of columns in 'y_score'" in str(e):
                print("Warning: Number of classes in y_true not equal to the number of columns in 'y_score'. Setting AUC-ROC to -999.")
            print("Warning: auc_roc_score() failed!")
            auc_roc = -999

        results = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'mcc': mcc,
            'auc_roc': auc_roc,
        }
        return results


def print_results(results, best_val_accuracy):
    print('--------------------- Results ----------------------')
    print('best_val_accuracy: {:.4f}'.format(best_val_accuracy))
    print('Test Accuracy: {:.4f}'.format(results['accuracy']))
    print('f1: {:.2f}'.format(results['f1']))
    print('precision: {:.2f}'.format(results['precision']))
    print('recall: {:.2f}'.format(results['recall']))
    print('mcc: {:.2f}'.format(results['mcc']))
    print('auc_roc: {:.2f}'.format(results['auc_roc']))
    print('----------------------------------------------------')