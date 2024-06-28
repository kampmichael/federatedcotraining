import numpy as np

#predictions is a list of numpy arrays
def getConsensusLabels(predictions, consensus_method):
    if predictions is None:
        return None
    if isinstance(predictions, list) and len(predictions) == 0:
        return None
    if predictions.size == 0:
        return None
    if consensus_method == 'majority':
        if(predictions.ndim == 3): #one hot encoded labels
            num_classes = predictions.shape[2]
            predsInt = np.argmax(predictions, axis=2).T 
            labels = np.zeros(predsInt.shape[0]).astype(int)
            for i in range(predsInt.shape[1]):
                labels[i] = np.bincount(predsInt[i]).argmax()
            labels_onehot = np.zeros((labels.size, num_classes))
            labels_onehot[np.arange(labels.size), labels] = 1
            return labels_onehot.astype(predictions.dtype)
        else: #labels encoded as int
            #print(predictions.shape)
            labels = np.zeros(predictions.shape[1])
            for i in range(predictions.shape[1]):
                #print(np.bincount(predictions[:,i]))
                labels[i] = np.bincount(predictions[:,i].astype(int)).argmax()
            return labels.astype(predictions.dtype)
    print("ERROR: consensus method ",consensus_method," unknown.")
    return None

    
def getConsensusLabelsQualified(predictions, consensus_method, agreement_threshold=0.8):
    if predictions is None:
        return None
    if isinstance(predictions, list) and len(predictions) == 0:
        return None
    if predictions.size == 0:
        return None
    
    if consensus_method == 'majority':
        num_clients, num_samples = len(predictions), len(predictions[0])
        num_classes = np.max([np.max(client_pred) for client_pred in predictions]) + 1

        final_predictions = []

        for sample_idx in range(num_samples):
            class_votes = np.zeros(num_classes)
            for client_pred in predictions:
                class_votes[client_pred[sample_idx]] += 1

            class_votes_percentage = class_votes / num_clients
            majority_label = np.argmax(class_votes_percentage)

            if class_votes_percentage[majority_label] >= agreement_threshold:
                final_predictions.append(majority_label)
            else:
                final_predictions.append(-1)  # Flag for disagreement

        return final_predictions

    print("ERROR: consensus method", consensus_method, "unknown.")
    return None

def getConsensusSoftDecisions(soft_decisions, consensus):
    if consensus == "average":
        return np.mean(soft_decisions, axis=0)
    elif consensus == "max":
        return np.max(soft_decisions, axis=0)
    elif consensus == "min":
        return np.min(soft_decisions, axis=0)
    else:
        raise ValueError("Invalid consensus method. Please choose 'average', 'max', or 'min'.")
