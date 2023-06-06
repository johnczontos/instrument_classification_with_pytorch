from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score, accuracy_score
import numpy as np

def calculate_metrics(predicted, scores, labels):
        accuracy = accuracy_score(labels, predicted)
        f1 = f1_score(labels, predicted, average='macro')
        precision = precision_score(labels, predicted, average='macro')
        recall = recall_score(labels, predicted, average='macro')
        mcc = matthews_corrcoef(labels, predicted)
        # auc_roc = roc_auc_score(scores, predicted, average='macro', multi_class='ovr')
        auc_roc = -999
        results = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'mcc': mcc,
            'auc_roc': auc_roc
        }
        return results