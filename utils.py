import torch
import config
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
DEVICE=config.DEVICE

def train_and_evaluate_one_epoch(model, criterion, optimizer, train_loader, val_loader):
    model.to(DEVICE)
    model.train()
    train_loss=0
    all_train_preds=[]
    all_train_labels=[]
    for images, labels, _ in train_loader:
        images=images.to(DEVICE)
        labels=labels.to(DEVICE)
        optimizer.zero_grad()
        train_outputs=model(images)
        loss=criterion(train_outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss +=loss.item()
        predictions=torch.argmax(train_outputs, dim=1)
        all_train_preds.extend(predictions.detach().cpu().tolist())
        all_train_labels.extend(labels.detach().cpu().tolist())
    train_loss /=len(train_loader)
    train_f1_macro=f1_score(all_train_labels, all_train_preds, average="macro", zero_division=0)
    train_f1_micro=f1_score(all_train_labels, all_train_preds, average="micro", zero_division=0)
    train_f1_weighted=f1_score(all_train_labels, all_train_preds, average="weighted", zero_division=0)
    
    model.eval()
    val_loss=0
    all_val_outputs=[]
    all_val_preds=[]
    all_val_labels=[]
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images=images.to(DEVICE)
            labels=labels.to(DEVICE)
            val_outputs=model(images)
            loss=criterion(val_outputs, labels)
            val_loss +=loss.item()
            predictions=torch.argmax(val_outputs, dim=1)
            all_val_outputs.extend(val_outputs.detach().cpu().tolist())
            all_val_preds.extend(predictions.detach().cpu().tolist())
            all_val_labels.extend(labels.detach().cpu().tolist())
    val_loss /=len(val_loader)
    val_accuracy=accuracy_score(all_val_labels, all_val_preds)
    val_precision_macro=precision_score(all_val_labels, all_val_preds, average="macro", zero_division=0)
    val_precision_micro=precision_score(all_val_labels, all_val_preds, average="micro", zero_division=0)
    val_precision_weighted=precision_score(all_val_labels, all_val_preds, average="weighted", zero_division=0)
    val_recall_macro=recall_score(all_val_labels, all_val_preds, average="macro", zero_division=0)
    val_recall_micro=recall_score(all_val_labels, all_val_preds, average="micro", zero_division=0)
    val_recall_weighted=recall_score(all_val_labels, all_val_preds, average="weighted", zero_division=0)
    val_f1_macro=f1_score(all_val_labels, all_val_preds, average="macro", zero_division=0)
    val_f1_micro=f1_score(all_val_labels, all_val_preds, average="micro", zero_division=0)
    val_f1_weighted=f1_score(all_val_labels, all_val_preds, average="weighted", zero_division=0)
    results_epoch={
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_f1_macro": train_f1_macro,
        "train_f1_micro": train_f1_micro,
        "train_f1_weighted": train_f1_weighted,
        "val_accuracy": val_accuracy,
        "val_precision_macro": val_precision_macro,
        "val_precision_micro": val_precision_micro,
        "val_precision_weighted": val_precision_weighted,
        "val_recall_macro": val_recall_macro,
        "val_recall_micro": val_recall_micro,
        "val_recall_weighted": val_recall_weighted,
        "val_f1_macro": val_f1_macro,
        "val_f1_micro": val_f1_micro,
        "val_f1_weighted": val_f1_weighted,
        "all_val_outputs": all_val_outputs,
        "all_val_preds": all_val_preds,
        "all_val_labels": all_val_labels,
    }
    return results_epoch, model

def evaluate_test(model, test_loader):
    model.to(DEVICE)
    model.eval()
    all_test_outputs=[]
    all_test_preds=[]
    all_test_labels=[]
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images=images.to(DEVICE)
            labels=labels.to(DEVICE)
            test_outputs=model(images)
            predictions=torch.argmax(test_outputs, dim=1)
            all_test_outputs.extend(test_outputs.detach().cpu().tolist())
            all_test_preds.extend(predictions.detach().cpu().tolist())
            all_test_labels.extend(labels.detach().cpu().tolist())
    test_accuracy=accuracy_score(all_test_labels, all_test_preds)
    test_precision_macro=precision_score(all_test_labels, all_test_preds, average="macro", zero_division=0)
    test_precision_micro=precision_score(all_test_labels, all_test_preds, average="micro", zero_division=0)
    test_precision_weighted=precision_score(all_test_labels, all_test_preds, average="weighted", zero_division=0)
    test_recall_macro=recall_score(all_test_labels, all_test_preds, average="macro", zero_division=0)
    test_recall_micro=recall_score(all_test_labels, all_test_preds, average="micro", zero_division=0)
    test_recall_weighted=recall_score(all_test_labels, all_test_preds, average="weighted", zero_division=0)
    test_f1_macro=f1_score(all_test_labels, all_test_preds, average="macro", zero_division=0)
    test_f1_micro=f1_score(all_test_labels, all_test_preds, average="micro", zero_division=0)
    test_f1_weighted=f1_score(all_test_labels, all_test_preds, average="weighted", zero_division=0)
    results={
        "test_accuracy": test_accuracy,
        "test_precision_macro": test_precision_macro,
        "test_precision_micro": test_precision_micro,
        "test_precision_weighted": test_precision_weighted,
        "test_recall_macro": test_recall_macro,
        "test_recall_micro": test_recall_micro,
        "test_recall_weighted": test_recall_weighted,
        "test_f1_macro": test_f1_macro,
        "test_f1_micro": test_f1_micro,
        "test_f1_weighted": test_f1_weighted,
        "all_test_outputs": all_test_outputs,
        "all_test_preds": all_test_preds,
        "all_test_labels": all_test_labels,
    }
    return results