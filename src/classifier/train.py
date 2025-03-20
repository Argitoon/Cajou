# 1. Built-in Python libraries
import os
import math
import shutil

# 2. Third-party libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

# 3. Project-specific imports
from classifier.dataloader import get_dataloader
from classifier.model import *
from classifier.tools import validate_path

with open("variables.yaml", "r", encoding="utf-8") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

MODEL_TYPE = config["MODEL_TYPE"]
MODEL_PATH = config["LOCAL_PATH"]["TRAIN_MODEL"]
DATA_PATH = config["LOCAL_PATH"]["TRAIN_DATASET"]
RESULTS_PATH = config["LOCAL_PATH"]["TRAIN_RESULTS"]

TEST_DATA_PATH = config["LOCAL_PATH"]["TEST_UNICLASS_DATASET"]

APP_MODEL_PATH = config["LOCAL_PATH"]["APP_MODEL"]
APP_DATA_PATH = config["LOCAL_PATH"]["APP_DATASET"]

def select_model(model_type : str = MODEL_TYPE) -> nn.Module :
    """
    Select the model to use for the classification
    
    :param model_type: The type of model to use (default ResNet50_TL for best results)
    
    :return: the model to use
    """
    if model_type == "ResNet50_TL":
        return ResNet50_TL
    elif model_type == "RegNet_TL":
        return RegNet_TL
    elif model_type == "CustomResNet":
        return CustomResNet
    elif model_type == "DenseNet121_TL":
        return DenseNet121_TL
    elif model_type == "EfficientNetB0_TL":
        return EfficientNetB0_TL
    elif model_type == "Xception_TL":
        return Xception_TL
    elif model_type == "ConvNeXT_TL":
        return ConvNeXT_TL
    else:
        raise ValueError(f"Model type {model_type} not recognized")

def train_model(model_type : str = MODEL_TYPE,
                model_path : str = MODEL_PATH, 
                trainval_dataset : str = DATA_PATH, 
                test_dataset : str = TEST_DATA_PATH,
                num_epochs : int = 51, batch_size : int = 8, 
                learning_rate : float = 0.001, step_save : int = 10) -> None :
    """
    Train the model with the given parameters and save the model at given path
    
    :param model_type: The type of model to use
    :param model_path: The path to save the model
    :param trainval_dataset: The path to the training (and validation) dataset
    :param test_dataset: The path to the test dataset (if None, the test dataset is the validation dataset)
    :param num_epochs: The number of epochs to train the model (default 51 for best results)
    :param batch_size: The batch size (default 8 because of memory limitations)
    :param learning_rate: The learning rate (default 0.001)
    :param step_save: The number of epochs between each save (default 10 arbitrarily)
    """
    
    # I. Load the data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader_train, dataloader_val, classes = get_dataloader(data_dir=trainval_dataset, batch_size=batch_size)
    
    if test_dataset is None:
        dataloader_test = dataloader_val
    else:
        dataloader_test, _, classes = get_dataloader(data_dir=test_dataset, batch_size=1000, augment_data=False, test_size=0.)
    
    # II. Initialize the model
    model = select_model(model_type)(num_classes=len(classes), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # III. Train the model
    nb_evals = math.ceil(num_epochs / step_save)
    epochs_list = [i for i in range(num_epochs) if i % step_save == 0]
    
    losses = {
        "train": [0] * nb_evals,
        "val": [0] * nb_evals,
        "test": [0] * nb_evals
    }
    accuracies = {
        "train": [0] * nb_evals,
        "val": [0] * nb_evals,
        "test": [0] * nb_evals
    }
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{num_epochs}") as tepoch:
            for images, labels in tepoch:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                tepoch.set_postfix(loss=running_loss / len(dataloader_train), accuracy=100. * correct / total, epoch=epoch+1)
        
        # Save Train and Validation Loss and Accuracy  
        if epoch % step_save == 0:
            model.eval()
            
            def evaluate(dataloader):
                """
                Evaluate the model on the given dataloader
                
                :param dataloader: the dataloader to evaluate (train, validation, test)
                """
                loss = 0.0
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in dataloader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()
                return loss / len(dataloader), 100. * correct / total
            
            losses["train"][epoch // step_save], accuracies["train"][epoch // step_save] = evaluate(dataloader_train)
            losses["val"][epoch // step_save], accuracies["val"][epoch // step_save] = evaluate(dataloader_val)
            losses["test"][epoch // step_save], accuracies["test"][epoch // step_save] = evaluate(dataloader_test)
            
    # IV. Save the model
    validate_path(model_path)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved under {model_path}")
    
    def trace_curves(curve_type, data):
        """
        Create the loss or accuracy curves
        """
        plt.figure(figsize=(6, 5))        
        for key, values in data.items():
            # Convert torch.Tensor to numpy array
            values = [v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for v in values]
            values = np.array(values)
            plt.plot(epochs_list, values, label=f"{key} {curve_type}")
            
        plt.ylabel(curve_type)
        plt.legend()
        plt.title(f"Evolution de {curve_type}")
        
        path = RESULTS_PATH + f"/{curve_type}_evolution_train.png"
        validate_path(path)
        plt.savefig(path)
        print(f"Graph of {curve_type} saved under '{path}'")
    
    trace_curves("loss", losses)
    trace_curves("accuracy", accuracies)

def main(option : int = 0) -> None:
    """
    Main for training the model
    
    :param option: the option to choose the type of training (default or application)
    """
    
    # I. Set parameters based on the option    
    if option == 0 : # Default training
        model_path = MODEL_PATH
        trainval_dataset = DATA_PATH
        test_dataset = TEST_DATA_PATH
    elif option == 1 : # Application training
        model_path = APP_MODEL_PATH
        trainval_dataset = APP_DATA_PATH
        test_dataset = TEST_DATA_PATH
        
        if not os.path.exists(trainval_dataset):
            raise FileNotFoundError(f"The folder {trainval_dataset} does not exist, use the application to start "
                             + "the training or create the folder and add images for all classes manually")        
    else:
        raise ValueError("Option not recognized")

    # II. Train the model
    train_model(MODEL_TYPE, model_path, trainval_dataset, test_dataset)
    
    # III. Clean the app/data folder (if any) 
    if option == 1: 
        if os.path.exists(APP_DATA_PATH):
            shutil.rmtree(APP_DATA_PATH)
    
if __name__ == "__main__":
    print("To train please got to ~/pfe/ and run \'python main.py train\'")