import numpy as np


# TP = predicted as OOD and true label is OOD
# TN = predicted as IN and true label is IN
# FP = predicted as OOD and true label is IN
# FN = predicted as IN and true label is OOD

# FAR = Number of accepted OOD sentences / Number of OOD sentences
# FAR = FN / (TP + FN)

# FRR = Number of rejected ID sentences / Number of ID sentences
# FRR = FP / (FP + TN)


class Testing:
    """Used to test the results of classification."""

    def __init__(self, model, X_test, y_test, model_name: str, oos_label, bin_model=None, bin_oos_label=None):
        self.model = model
        self.X_test = X_test  # tf.Tensor
        self.y_test = y_test  # tf.Tensor
        self.oos_label = oos_label  # number
        self.model_name = model_name
        self.bin_model = bin_model
        self.bin_oos_label = bin_oos_label

    def test_train(self):
        accuracy_correct, accuracy_out_of = 0, 0
        recall_correct, recall_out_of = 0, 0

        tp, tn, fp, fn = 0, 0, 0, 0

        pred_labels = self.model.predict(self.X_test)

        for pred_label, true_label in zip(pred_labels, self.y_test):

            # the following set of conditions is the same for all testing methods
            if true_label != self.oos_label:
                if pred_label == true_label:
                    accuracy_correct += 1

                if pred_label != self.oos_label:
                    tn += 1
                else:
                    fp += 1

                accuracy_out_of += 1
            else:
                if pred_label == true_label:
                    recall_correct += 1
                    tp += 1
                else:
                    fn += 1

                recall_out_of += 1

        accuracy = accuracy_correct / accuracy_out_of * 100
        recall = recall_correct / recall_out_of * 100

        far = fn / (tp + fn) * 100  # false acceptance rate
        frr = fp / (fp + tn) * 100  # false recognition rate

        return {'accuracy': round(accuracy, 1), 'recall': round(recall, 1), 'far': round(far, 1), 'frr': round(frr, 1)}

    def test_threshold(self, threshold: float):
        accuracy_correct, accuracy_out_of = 0, 0
        recall_correct, recall_out_of = 0, 0

        tp, tn, fp, fn = 0, 0, 0, 0

        pred_probs = self.model.predict_proba(
            self.X_test)  # returns numpy array

        pred_labels = np.argmax(pred_probs, axis=1)
        pred_similarities = np.take_along_axis(pred_probs, indices=np.expand_dims(pred_labels, axis=1),
                                               axis=1).squeeze()

        for pred_label, pred_similarity, true_label in zip(pred_labels, pred_similarities, self.y_test):
            if pred_similarity < threshold:
                pred_label = self.oos_label

            # the following set of conditions is the same for all testing methods
            if true_label != self.oos_label:
                if pred_label == true_label:
                    accuracy_correct += 1

                if pred_label != self.oos_label:
                    tn += 1
                else:
                    fp += 1

                accuracy_out_of += 1
            else:
                if pred_label == true_label:
                    recall_correct += 1
                    tp += 1
                else:
                    fn += 1

                recall_out_of += 1

        accuracy = accuracy_correct / accuracy_out_of * 100
        recall = recall_correct / recall_out_of * 100

        far = fn / (tp + fn) * 100  # false acceptance rate
        frr = fp / (fp + tn) * 100  # false recognition rate

        return {'accuracy': round(accuracy, 1), 'recall': round(recall, 1), 'far': round(far, 1), 'frr': round(frr, 1)}

    def test_binary(self):
        accuracy_correct, accuracy_out_of = 0, 0
        recall_correct, recall_out_of = 0, 0

        tp, tn, fp, fn = 0, 0, 0, 0

        pred_bin_labels = self.bin_model.predict(self.X_test)
        pred_multi_labels = self.model.predict(self.X_test)

        for pred_bin_label, pred_multi_label, true_label in zip(pred_bin_labels, pred_multi_labels, self.y_test):
            if pred_bin_label != self.bin_oos_label:
                pred_label = pred_multi_label
            else:
                pred_label = self.oos_label

            # the following set of conditions is the same for all testing methods
            if true_label != self.oos_label:
                if pred_label == true_label:
                    accuracy_correct += 1

                if pred_label != self.oos_label:
                    tn += 1
                else:
                    fp += 1

                accuracy_out_of += 1
            else:
                if pred_label == true_label:
                    recall_correct += 1
                    tp += 1
                else:
                    fn += 1

                recall_out_of += 1

        accuracy = accuracy_correct / accuracy_out_of * 100
        recall = recall_correct / recall_out_of * 100

        far = fn / (tp + fn) * 100  # false acceptance rate
        frr = fp / (fp + tn) * 100  # false recognition rate

        return {'accuracy': round(accuracy, 1), 'recall': round(recall, 1), 'far': round(far, 1), 'frr': round(frr, 1)}
