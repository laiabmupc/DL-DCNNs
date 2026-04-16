import os
import json
import torch
import torch.nn as nn
import config
import copy
import itertools
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import MAMeDataset
from resnet import Resnet
from standard_architecture import PyramidalAlternatingPooling, HourglassAlternatingPooling
from convnextv2 import convnextv2_femto, convnextv2_nano, convnextv2_tiny
from utils import train_and_evaluate_one_epoch, evaluate_test

CONVNETXV2_PRETRAINED={"femto": "/gpfs/home/nct/nct01188/DL-lab/pretrained_weights/convnextv2_femto_1k_224_fcmae.pt",
                        "nano": "/gpfs/home/nct/nct01188/DL-lab/pretrained_weights/convnextv2_nano_1k_224_fcmae.pt",
                        "tiny": "/gpfs/home/nct/nct01188/DL-lab/pretrained_weights/convnextv2_tiny_1k_224_fcmae.pt"}
EPOCHS=20
G=torch.Generator()
G.manual_seed(32)
DEVICE=config.DEVICE

def train_val_test_dataloaders(data_aug=False, batch_size=32, transform=None, num_imgs=0):
    if data_aug:
        train_set=MAMeDataset(config.TRAIN_PATH, transform=transform, num_imgs=num_imgs, data_aug=True)
    else:
        train_set=MAMeDataset(config.TRAIN_PATH, transform=transform, num_imgs=num_imgs)

    val_set=MAMeDataset(config.VAL_PATH, transform=transform, num_imgs=0)
    test_set=MAMeDataset(config.TEST_PATH, transform=transform, num_imgs=0)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=G)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
    
def run_train_and_eval(model, model_name, criterion, optimizer, train_loader, val_loader, test_loader):
    original_model=copy.deepcopy(model)
    results_val={}
    best_epoch_loss=-1
    best_loss=float("inf")
    best_epoch_f1=-1
    best_f1=float("-inf")

    for epoch in range(EPOCHS):
        results_epoch, model_train=train_and_evaluate_one_epoch(model, criterion, optimizer, train_loader, val_loader)
        results_val[f"Epoch: {epoch}"]=results_epoch
        print(f"Epoch: {epoch} | Train Loss={results_epoch['train_loss']:.4f} | Val Loss={results_epoch['val_loss']:.4f} | Val Acc={results_epoch['val_accuracy']:.4f} | Val F1 macro={results_epoch['val_f1_macro']:.4f}")
        if results_epoch["val_loss"] < best_loss:
            best_loss=results_epoch["val_loss"]
            best_epoch_loss=epoch
            torch.save(model_train.state_dict(), os.path.join(config.RESULTS, f"{model_name}_best_loss.pt"))
        if results_epoch["val_f1_macro"] > best_f1:
            best_f1=results_epoch["val_f1_macro"]
            best_epoch_f1=epoch
            torch.save(model_train.state_dict(), os.path.join(config.RESULTS, f"{model_name}_best_f1.pt"))
    
    with open(os.path.join(config.RESULTS, f"{model_name}_val_results.json"), "w") as f:
        json.dump(results_val, f)
    results_test={}
    
    best_model_loss_path=os.path.join(config.RESULTS, f"{model_name}_best_loss.pt")
    best_model_f1_path=os.path.join(config.RESULTS, f"{model_name}_best_f1.pt")
    model_loss=copy.deepcopy(original_model)
    model_loss.load_state_dict(torch.load(best_model_loss_path, map_location=config.DEVICE))
    res_loss_test=evaluate_test(model_loss, test_loader)
    res_loss_test["epoch"]=best_epoch_loss
    model_f1=copy.deepcopy(original_model)
    model_f1.load_state_dict(torch.load(best_model_f1_path, map_location=config.DEVICE))
    res_f1_test=evaluate_test(model_f1, test_loader)
    res_f1_test["epoch"]=best_epoch_f1
    results_test["best_val_loss"]=res_loss_test
    results_test["best_val_f1_macro"]=res_f1_test
    
    with open(os.path.join(config.RESULTS, f"{model_name}_test_results.json"), "w") as f:
        json.dump(results_test, f)
    return results_val, results_test

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

def run_standard_underfitting():
    transforms_=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])
    train_loader, val_loader, test_loader=train_val_test_dataloaders(data_aug=False, batch_size=64, transform=transforms_, num_imgs=0) # High batch size
    model=PyramidalAlternatingPooling(num_classes=29, in_channels=3, dropout_rate=0.5) # High dropout for underfitting
    criterion=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-3) # Low learning rate for underfitting
    model_name="standard_bs_64_im_0_drop_0.5_adamlr_1e-6_wdecay_1e-3_underfitting"
    results_val, results_test=run_train_and_eval(model, model_name, criterion, optimizer, train_loader, val_loader, test_loader)
    
def run_standard_overfitting():
    transforms_=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])
    train_loader, val_loader, test_loader=train_val_test_dataloaders(data_aug=False, batch_size=16, transform=transforms_, num_imgs=300) # Low batch size
    model=PyramidalAlternatingPooling(num_classes=29, in_channels=3, dropout_rate=0) # Tiny and No dropout for overfitting
    criterion=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0) # Higher learning rate for overfitting
    model_name="standard_bs_16_im_300_drop_0_adamlr_1e-3_wdecay_0_overfitting"
    results_val, results_test=run_train_and_eval(model, model_name, criterion, optimizer, train_loader, val_loader, test_loader)

def run_standard_fine_tuning():
    transforms_=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])
    # Hyperparams grid
    opt_names=["adamw"]
    learning_rates=[1e-3, 1e-5]
    drop_path_rates=[0.0, 0.3]
    batch_sizes=[16, 32]
    data_augmentation=[True, False]
    weight_decay=[0, 1e-4]
    weight_init=[True, False]
    all_configs=list(itertools.product(opt_names, learning_rates, drop_path_rates, batch_sizes, weight_decay, data_augmentation, weight_init))
    print(f"Total experiments: {len(all_configs)}")
    for i, (opt, lr, dropout, bs, wd, da, wi) in enumerate(all_configs):
        print(f"Experiment {i+1}/{len(all_configs)}")
        model_name=f"pyramidal_{opt}_{lr}_{dropout}_{bs}_{wd}_{da}_{wi}"
        train_loader, val_loader, test_loader=train_val_test_dataloaders(data_aug=da, batch_size=bs, transform=transforms_, num_imgs=0)
        model=PyramidalAlternatingPooling(num_classes=29, in_channels=3, dropout_rate=dropout)
        if wi:
           model.apply(initialize_weights) 
        model=model.to(DEVICE)
        criterion=torch.nn.CrossEntropyLoss()
        optimizer=get_optimizer(opt, model, lr, weight_decay=wd)
        results_val, results_test=run_train_and_eval(model, model_name, criterion, optimizer, train_loader, val_loader, test_loader)

def run_resnet_underfitting():
    transforms_=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])
    train_loader, val_loader, test_loader=train_val_test_dataloaders(data_aug=False, batch_size=64, transform=transforms_, num_imgs=0) # High batch size
    model=Resnet(num_classes=29, in_channels=3, dropout_rate=0.5) # High dropout for underfitting
    criterion=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-3) # Low learning rate for underfitting
    model_name="resnet_bs_64_im_0_drop_0.5_adamlr_1e-6_wdecay_1e-3_underfitting"
    results_val, results_test=run_train_and_eval(model, model_name, criterion, optimizer, train_loader, val_loader, test_loader)

def run_resnet_overfitting():
    transforms_=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])
    train_loader, val_loader, test_loader=train_val_test_dataloaders(data_aug=False, batch_size=16, transform=transforms_, num_imgs=500) # Low batch size
    model=Resnet(num_classes=29, in_channels=3, dropout_rate=0) # Tiny and No dropout for overfitting
    criterion=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0) # Higher learning rate for overfitting
    model_name="resnet_bs_16_im_500_drop_0_adamlr_1e-3_wdecay_0_overfitting"
    results_val, results_test=run_train_and_eval(model, model_name, criterion, optimizer, train_loader, val_loader, test_loader)

def run_resnet_fine_tuning():
    transforms_=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])
    # Hyperparams grid
    opt_names=["adamw"]
    learning_rates=[1e-3, 1e-5]
    drop_path_rates=[0.0, 0.3]
    batch_sizes=[16, 32]
    data_augmentation=[True, False]
    weight_decay=[0, 1e-4]
    weight_init=[True, False]
    all_configs=list(itertools.product(opt_names, learning_rates, drop_path_rates, batch_sizes, weight_decay, data_augmentation, weight_init))
    print(f"Total experiments: {len(all_configs)}")
    for i, (opt, lr, dropout, bs, wd, da, wi) in enumerate(all_configs):
        print(f"Experiment {i+1}/{len(all_configs)}")
        model_name=f"resnet_{opt}_{lr}_{dropout}_{bs}_{wd}_{da}_{wi}"
        train_loader, val_loader, test_loader=train_val_test_dataloaders(data_aug=da, batch_size=bs, transform=transforms_, num_imgs=0)
        model=Resnet(num_classes=29, in_channels=3, dropout_rate=dropout)
        if wi:
           model.apply(initialize_weights) 
        model=model.to(DEVICE)
        criterion=torch.nn.CrossEntropyLoss()
        optimizer=get_optimizer(opt, model, lr, weight_decay=wd)
        results_val, results_test=run_train_and_eval(model, model_name, criterion, optimizer, train_loader, val_loader, test_loader)

def run_convnextv2_underfitting():
    transforms_=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])
    train_loader, val_loader, test_loader=train_val_test_dataloaders(data_aug=False, batch_size=64, transform=transforms_, num_imgs=1000) #High batch size and low number of images for underfitting
    model=convnextv2_femto(in_chans=3, num_classes=29, drop_path_rate=0.9) #Femto and High dropout for underfitting
    criterion=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-10, weight_decay=0.7) #Low learning rate for underfitting
    model_name="convnextv2_femto_bs_64_im_1000_drop_0.9_adamlr_1e-10_underfitting"
    results_val, results_test=run_train_and_eval(model, model_name, criterion, optimizer, train_loader, val_loader, test_loader)

def run_convnextv2_overfitting():
    transforms_=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])
    train_loader, val_loader, test_loader=train_val_test_dataloaders(data_aug=False, batch_size=16, transform=transforms_, num_imgs=10000) #Low batch size and 10000 images for overfitting
    model=convnextv2_tiny(in_chans=3, num_classes=29, drop_path_rate=0) #Tiny and No dropout for overfitting
    #Load pretrained weights for overfitting
    state_dict=torch.load(CONVNETXV2_PRETRAINED["tiny"], map_location="cpu")
    state_dict=state_dict["model"]
    state_dict.pop("head.weight", None)
    state_dict.pop("head.bias", None)
    for key in list(state_dict.keys()):
        if key.endswith("grn.gamma") or key.endswith("grn.beta"):
            if state_dict[key].ndim==2:
                state_dict[key]=state_dict[key].unsqueeze(1).unsqueeze(1)
    model.load_state_dict(state_dict, strict=False)
    criterion=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0) #Higher learning rate for overfitting
    model_name="convnextv2_tiny_bs_16_im_10000_drop_0_adamwlr_1e-4_overfitting"
    results_val, results_test=run_train_and_eval(model, model_name, criterion, optimizer, train_loader, val_loader, test_loader)

def get_convnextv2_model(model_size, num_classes=29, in_chans=3, drop_path_rate=0, pretrained=False):
    if model_size=="femto":
        model=convnextv2_femto(in_chans=in_chans, num_classes=num_classes, drop_path_rate=drop_path_rate)
    elif model_size=="nano":
        model=convnextv2_nano(in_chans=in_chans, num_classes=num_classes, drop_path_rate=drop_path_rate)
    elif model_size=="tiny":
        model=convnextv2_tiny(in_chans=in_chans, num_classes=num_classes, drop_path_rate=drop_path_rate)
    else:
        raise ValueError(f"Model not supported: {model_size}")
    if pretrained:
        state_dict=torch.load(CONVNETXV2_PRETRAINED[model_size], map_location="cpu")
        state_dict=state_dict["model"]
        state_dict.pop("head.weight", None)
        state_dict.pop("head.bias", None)
        for key in list(state_dict.keys()):
            if key.endswith("grn.gamma") or key.endswith("grn.beta"):
                if state_dict[key].ndim==2:
                    state_dict[key]=state_dict[key].unsqueeze(1).unsqueeze(1)
        model.load_state_dict(state_dict, strict=False)
    return model

def get_optimizer(optimizer_name, model, lr, weight_decay=0):
    optimizer_name=optimizer_name.lower()
    if optimizer_name=="adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name=="adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer not supported: {optimizer_name}")

def run_convnextv2_fine_tuning():
    transforms_=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])
    #Hyperparams grid
    model_sizes=["femto", "nano", "tiny"]
    opt_names=["adamw"]
    learning_rates=[1e-3, 1e-5]
    pretrained=[True]
    drop_path_rates=[0.0, 0.3]
    batch_sizes=[16]
    data_augmentation=[True, False]
    weight_decay=[0, 1e-4]
    all_configs=list(itertools.product(model_sizes, opt_names, learning_rates, pretrained, drop_path_rates, batch_sizes, data_augmentation, weight_decay))
    all_configs=all_configs[:16]
    print(f"Total experiments: {len(all_configs)}")
    for i, (model_size, opt, lr, weightspretrained, dropout, bs, da, wd) in enumerate(all_configs):
        print(f"Experiment {i+1}/{len(all_configs)}")
        model_name=f"convnextv2_{model_size}_{opt}_{lr}_{weightspretrained}_{dropout}_{bs}_{da}_{wd}"
        train_loader, val_loader, test_loader=train_val_test_dataloaders(data_aug=da, batch_size=bs, transform=transforms_, num_imgs=0)
        model=get_convnextv2_model(model_size, num_classes=29, in_chans=3, drop_path_rate=dropout, pretrained=weightspretrained)
        model=model.to(DEVICE)
        criterion=torch.nn.CrossEntropyLoss()
        optimizer=get_optimizer(opt, model, lr, weight_decay=wd)
        results_val, results_test=run_train_and_eval(model, model_name, criterion, optimizer, train_loader, val_loader, test_loader)
    
RUNS={
    "standard":{
        "underfitting":run_standard_underfitting,
        "overfitting": run_standard_overfitting,
        "fine_tuning": run_standard_fine_tuning,
    },
    "resnet":{
        "underfitting":run_resnet_underfitting,
        "overfitting": run_resnet_overfitting,
        "fine_tuning": run_resnet_fine_tuning,
    },
    "convnextv2":{
        "underfitting":run_convnextv2_underfitting,
        "overfitting": run_convnextv2_overfitting,
        "fine_tuning": run_convnextv2_fine_tuning,
    },
}
def main(model="resnet", run="underfitting"):
    try:
        RUNS[model][run]()
    except KeyError:
        raise ValueError(f"No valid configuration: model={model}, run={run}")

if __name__=="__main__":
    # main()
    torch.cuda.empty_cache()
    run_convnextv2_fine_tuning()

    # for run in ["fine_tuning", "overfitting", "underfitting"]:
    #     torch.cuda.empty_cache()
    #     main(model='resnet', run=run)
    