import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def softmax(x):
    x= np.asarray(x, dtype=np.float64)
    x= x-np.max(x, axis=1, keepdims=True)
    exp_x= np.exp(x)
    return exp_x/np.sum(exp_x, axis=1, keepdims=True)

def outputs_bestval_epoch(val_json_path, selection_metric="val_f1_macro"):
    with open(val_json_path, "r", encoding="utf-8") as f:
        data= json.load(f)
    best_epoch= None
    best_value= float("-inf")
    for epoch_name, metrics in data.items():
        value= metrics.get(selection_metric, None)
        if value is not None and value>best_value:
            best_value= value
            best_epoch= epoch_name
    best_metrics= data[best_epoch]
    val_outputs= np.array(best_metrics["all_val_outputs"],dtype=np.float32)
    val_labels= np.array(best_metrics["all_val_labels"],dtype=np.int64)
    return val_outputs, val_labels, best_epoch, best_value


def load_test_outputs(test_json_path, key="best_val_f1_macro"):
    with open(test_json_path, "r", encoding="utf-8") as f:
        data= json.load(f)
    test_data= data[key]
    test_outputs= np.array(test_data["all_test_outputs"],dtype=np.float32)
    test_labels= np.array(test_data["all_test_labels"],dtype=np.int64)
    return test_outputs, test_labels

def build_features(outputs_model_1, outputs_model_2, use_probabilities=True):
    if use_probabilities:
        feat_1= softmax(outputs_model_1)
        feat_2= softmax(outputs_model_2)
    else:
        feat_1= outputs_model_1
        feat_2= outputs_model_2
    return np.concatenate([feat_1, feat_2], axis=1)

def evaluate_predictions(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred),"precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),"recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),"f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),"f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),"f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),}


def run_log_ensemble(model1_val_json,model1_test_json,model2_val_json,model2_test_json,selection_metric="val_f1_macro",test_key="best_val_f1_macro",use_probabilities=True):
    val_outputs1, val_labels1, best_epoch1, best_score1= outputs_bestval_epoch(model1_val_json, selection_metric=selection_metric)
    val_outputs2, val_labels2, best_epoch2, best_score2= outputs_bestval_epoch(model2_val_json, selection_metric=selection_metric)
    X_train= build_features(val_outputs1, val_outputs2, use_probabilities=use_probabilities)
    y_train= val_labels1
    if not np.array_equal(val_labels1, val_labels2):
        raise ValueError("The val labels do not match.")

    test_outputs1,test_labels1= load_test_outputs(model1_test_json, key=test_key)
    test_outputs2,test_labels2= load_test_outputs(model2_test_json, key=test_key)
    X_test= build_features(test_outputs1, test_outputs2, use_probabilities=use_probabilities)
    y_test= test_labels1
    if not np.array_equal(test_labels1, test_labels2):
        raise ValueError("The test labels do not match.")


    model= LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)
    print("log_ensemble")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
    return model, metrics, y_pred

def run_average_ensemble(model1_test_json,model2_test_json,test_key="best_val_f1_macro"):
    test_outputs1,test_labels1= load_test_outputs(model1_test_json, key=test_key)
    test_outputs2,test_labels2= load_test_outputs(model2_test_json, key=test_key)
    if not np.array_equal(test_labels1, test_labels2):
        raise ValueError("The test labels do not match.")
    probs1= softmax(test_outputs1)
    probs2= softmax(test_outputs2)
    avg_probs = (probs1 + probs2) / 2.0
    y_pred= np.argmax(avg_probs, axis=1)
    y_test= test_labels1
    metrics = evaluate_predictions(y_test, y_pred)
    print("Average ensemble")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    return metrics,y_pred,avg_probs

if __name__ == "__main__":
    model1_val_json= "/home/natalia/Escritorio/MAI/DL/DL/results/convnextv2_ensemble_val_results.json"
    model1_test_json= "/home/natalia/Escritorio/MAI/DL/DL/results/convnextv2_ensemble_test_results.json"

    model2_val_json= "/home/natalia/Escritorio/MAI/DL/DL/results/resnet_ensemble_val_results.json"
    model2_test_json= "/home/natalia/Escritorio/MAI/DL/DL/results/resnet_ensemble_test_results.json"

    model,metrics,y_pred =run_log_ensemble(model1_val_json=model1_val_json,model1_test_json=model1_test_json, model2_val_json=model2_val_json, model2_test_json=model2_test_json,selection_metric="val_f1_macro",test_key="best_val_f1_macro",use_probabilities=True)
    metrics,y_pred,avg_probs =run_average_ensemble(model1_test_json=model1_test_json, model2_test_json=model2_test_json,test_key="best_val_f1_macro")
