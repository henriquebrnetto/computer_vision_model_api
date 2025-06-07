def evaluate_model(predictions):    

    y_pred = predictions['pred'].astype(int)
    y_true = predictions['y_true'].astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    total_evaluated = tp + fp + tn + fn
    accuracy = 0.0
    if total_evaluated > 0:
        accuracy = float((tp + tn) / total_evaluated)

    precision_fraud = 0.0
    if (tp + fp) > 0:
        precision_fraud = float(tp / (tp + fp))
    
    recall_fraud = 0.0
    if (tp + fn) > 0:
        recall_fraud = float(tp / (tp + fn))

    # --- Calculate F1 Score for Fraud Class ---
    f1_score_fraud = 0.0
    if (precision_fraud + recall_fraud) > 0: # Avoid division by zero if both are 0
        f1_score_fraud = float(2 * (precision_fraud * recall_fraud) / (precision_fraud + recall_fraud))
    # --- End F1 Score Calculation ---

    evaluation_metrics = {
        "accuracy": accuracy,
        "precision_fraud": precision_fraud,
        "recall_fraud": recall_fraud,
        "f1_score_fraud": f1_score_fraud, # Add F1 score here
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total_evaluated": total_evaluated
    }

    return evaluation_metrics