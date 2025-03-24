# 1. Built-in Python libraries
import os
import yaml
from typing import Tuple, List, Union
import shutil

# 2. Third-party libraries
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 3. Project-specific imports
from classifier.model import *
from classifier.dataloader import transform_train as transform

with open("variables.yaml", "r", encoding="utf-8") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

MODEL_TYPE = config["MODEL_TYPE"]
CLASS_NAMES = config["CLASS_NAMES"]
MODEL_PATH = config["LOCAL_PATH"]["TEST_MODEL"]
DATA_PATH = config["LOCAL_PATH"]["TEST_UNICLASS_DATASET"]
RESULTS_PATH = config["LOCAL_PATH"]["TEST_RESULTS"]

def shorten_filename(filename: str, max_length: int = 30) -> str:
    """
    Raccourcit un nom de fichier trop long en ajoutant '...' à la fin.

    :param filename: Nom du fichier à raccourcir
    :param max_length: Longueur maximale autorisée
    :return: Nom de fichier raccourci
    """
    if len(filename) <= max_length:
        return filename

    return filename[:max_length - 3] + "..." 

def validate_path(path : str) -> str:
    """
    Validate a path
    
    :param path: the path to validate
    """
    
    components = path.split("/")
    for i in range(1, len(components) + 1):
        p = "/".join(components[:i])
        if not os.path.exists(p) and not '.' in components[i-1] :
            os.makedirs(p)
            print(f"Path {p} created")

def load_nb_classes(model_path : str = MODEL_PATH) -> int:
    """
    Find the number of classes in a model
    
    :param model_path: the path to the model
    """
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    
    for key in checkpoint.keys():
        if "fc.weight" in key or "classifier.weight" in key:
            num_classes = checkpoint[key].shape[0]
            break
        
    if num_classes is None:
        print("Error: Could not detect num_classes in the checkpoint.")
        return len(CLASS_NAMES) # Not always true, but better than nothing
    
    return num_classes

def load_model(model_type = MODEL_TYPE, 
               model_path : str = MODEL_PATH, 
               num_classes : int = None) -> nn.Module:
    """
    Load a model from a given path
    
    :param model_type: the type of model to load
    :param model_path: path to the model
    :param num_classes: number of classes in the model (if None, tries to detect it)
    
    :return: the model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if num_classes is None:
        num_classes = load_nb_classes(model_path)
    
    if model_type == "ResNet50_TL":
        model = ResNet50_TL(num_classes=num_classes)
    elif model_type == "DenseNet121_TL":
        model = DenseNet121_TL(num_classes=num_classes)
    elif model_type == "EfficientNetB0_TL":
        model = EfficientNetB0_TL(num_classes=num_classes)
    elif model_type == "RegNet_TL":
        model = RegNet_TL(num_classes=num_classes)
    elif model_type == "Xception_TL":
        model = Xception_TL(num_classes=num_classes)
    elif model_type == "ConvNeXT_TL":
        model = ConvNeXT_TL(num_classes=num_classes)
    elif model_type == "CustomResNet":
        model = CustomResNet(num_classes=num_classes)
    else:
        print(f"{model_type} does not exist."); exit()
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()
    return model

def predict_image(image_path : Union[str, Image.Image],
                  model, transform : transforms.Compose,
                  device : torch.device, top : int = 3) -> Tuple[List[int], List[float]] :
    """
    Predict the class of an image
    
    :param image_path: path to the image or image itself
    :param model: the model
    :param transform: the transformation to apply to the image
    :param device: the device to use
    :param top: the number of top classes to return
    
    :return: tuple of a list of the top classes and a list of the top probabilities    
    """
    
    if isinstance(image_path, str):
        image = Image.open(image_path)
    else :
        image = image_path
        
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    image = transform(image).unsqueeze(0).to(device)
    outputs = model(image)
    
    probabilities = torch.softmax(outputs, dim=1)
    top_prob, top_idx = probabilities.topk(top, dim=1)
    
    return top_idx.squeeze(0).tolist(), top_prob.squeeze(0).tolist()

def predict(model, image_path : str, top : int = 3) -> Tuple[List[str], List[float]]:
    """
    predict the class of an image
    
    :param model: the model
    :param image_path: path to the image
    :param top: the number of top classes to return
    
    :return: tuple of a list of the top classes and a list of the top probabilities
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    top_classes, top_probs = predict_image(image_path, model, transform, device, top=top)
    
    classes = CLASS_NAMES
    
    top_classes = [classes[i] for i in top_classes]

    return top_classes, top_probs

# Multiclass functions
def calculate_accuracy_multiclass(model, device, dataset : List[str]) -> float:
    """
    Calculate the accuracy of the model on a dataset, where images can belong to multiple classes
    
    :param model: the model
    :param device: the device
    :param dataset: the dataset (list of tuples of image path and accepted classes)
    
    :return: the accuracy
    """
    model.eval()
    
    correct = 0
    total = 0
    
    with tqdm(total=len(dataset), desc="Calculating accuracy") as pbar:
        for image_path, accepted_classes in dataset:
            pbar.update(1)
                      
            image = Image.open(image_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)
            
            output = model(image)
            
            _, predicted = output.max(1)

            predicted = predicted.item()
            total += 1
            
            if CLASS_NAMES[predicted] in accepted_classes:
                correct += 1
                
    #print(f"Correct: {correct}, Total: {total}")
    return correct / total * 100

def calculate_confidence_multiclass(model, device, dataset) -> Tuple[float, List[float]]:
    """
    Calculate the confidence of the model on a dataset, where images can belong to multiple classes
    
    :param model: the model
    :param dataloader: the dataloader
    
    :return: the confidence and the class confidences
    """
    model.eval()
    
    class_confidences = [0] * len(CLASS_NAMES)
    class_counts = [0] * len(CLASS_NAMES)
    confidence = 0
    
    with tqdm(total=len(dataset), desc="Calculating confidence") as pbar:
        for image_path, accepted_classes in dataset:
            pbar.update(1)
            
            image = Image.open(image_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)
            
            output = model(image)
            
            probs = F.softmax(output, dim=1)
            
            accepted_probs = 0
            
            for i, aclass in enumerate(accepted_classes):
                j = CLASS_NAMES.index(aclass)
                class_confidences[j] += probs[0][j].item()
                accepted_probs += probs[0][j].item()
                class_counts[j] += 1
            
            confidence += accepted_probs
    
    for i in range(len(class_confidences)):
        if class_counts[i] > 0:
            class_confidences[i] /= class_counts[i]
        
    return confidence / len(dataset), class_confidences    

def calculate_confusion_matrix_multiclass(model, device, dataset : List[str], save : bool = True,
                                save_path : str = RESULTS_PATH + "/confusion_matrix.png") -> None:
    """
    Calculate and save the confusion matrix for a dataset where images can belong to multiple classes
    
    :param model: the model
    :param device: the device
    :param dataset: the dataset
    :param save: whether to save the confusion matrix
    :param save_path: the path to save the confusion matrix
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with tqdm(total=len(dataset), desc="Calculating confusion matrix") as pbar:
        for image_path, accepted_classes in dataset:
            pbar.update(1)
            
            image = Image.open(image_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)
            
            output = model(image)
            
            _, predicted = output.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.append(CLASS_NAMES.index(accepted_classes[0]))
            
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap='Blues')
    plt.xticks(rotation=45)
    plt.title("Confusion Matrix")
    
    if save:
        validate_path(save_path)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Confusion matrix saved under {save_path}")
        
    plt.show()
    plt.clf()

# Uniclass functions
def calculate_accuracy_uniclass(model, device, dataloader : DataLoader) -> float:
    """
    Calculate the accuracy of the model on a dataloader where images belong to a single class
    
    :param model: the model
    :param device: the device
    :param dataloader: the dataloader
    
    :return: the accuracy
    """
    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(total=len(dataloader), desc="Calculating accuracy") as pbar:
            for images, labels in dataloader :
                pbar.update(1)
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
    return (100 * correct / total)

def calculate_confidence_uniclass(model, device, dataloader : DataLoader) -> Tuple[float, List[float]]:
    """
    Calculate the confidence of the model on a dataloader where images belong to a single class
    
    :param model: the model
    :param device: the device
    :param dataloader: the dataloader
    
    :return: tuple of the average confidence and the confidence for each class
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    class_confidences = [0] * len(CLASS_NAMES)
    class_counts = [0] * len(CLASS_NAMES)
    
    with torch.no_grad():
        with tqdm(total=len(dataloader), desc="Calculating confidence") as pbar:
            for images, labels in dataloader:
                pbar.update(1)
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                true_label_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
                
                for i in range(len(labels)):
                    class_idx = labels[i].item()
                    class_confidences[class_idx] += true_label_probs[i].item()
                    class_counts[class_idx] += 1
                
    for i in range(len(class_confidences)):
        if class_counts[i] > 0:
            class_confidences[i] /= class_counts[i]
            
    class_confidences_np = np.array(class_confidences)
    average_confidence = np.mean(class_confidences_np)
    return average_confidence, class_confidences

def calculate_confusion_matrix_uniclass(model, device, dataloader : DataLoader, save : bool = True,
                                save_path : str = RESULTS_PATH + "/confusion_matrix.png") -> None:
    """
    Calculate and save the confusion matrix for a dataloader where images belong to a single class
    
    :param model: the model
    :param device: the device
    :param dataloade: the dataloader
    :param save: whether to save the confusion matrix
    :param save_path: the path to save the confusion matrix
    """
    
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        with tqdm(total=len(dataloader), desc="Calculating confusion matrix") as pbar:
            for images, labels in dataloader:
                pbar.update(1)
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap='Blues')
    plt.xticks(rotation=45)
    plt.title("Confusion Matrix")
    
    if save:
        validate_path(save_path)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Confusion matrix saved under {save_path}")
    plt.show()
    plt.clf()
 
def main() -> None:
    # Some tests to check the functions
    
    # Shorten filename
    print(shorten_filename("This_is_a_very_very_very_very_very_long_filename.jpg"))
    
    # validate path
    test_path = "yahoooo/test/test/test/test"
    validate_path(test_path)
    if os.path.exists(test_path):
        print("Path validation successful")
        shutil.rmtree("yahoooo")
    
    print(load_nb_classes())
    print("Model loaded" if load_model() else "Model not loaded")
    
if __name__ == "__main__":
    print("To train please got to ~/pfe/ and run \'python main.py tools\'")