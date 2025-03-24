# 1. Built-in Python libraries
import os
import yaml

# 2. Third-party libraries
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader

# 3. Project-specific imports
from classifier.model import *
from classifier.dataloader import get_dataloader
from classifier.tools import (
    load_model, predict, validate_path,
    calculate_accuracy_uniclass, calculate_confidence_uniclass, calculate_confusion_matrix_uniclass,
    calculate_accuracy_multiclass, calculate_confidence_multiclass, calculate_confusion_matrix_multiclass
)

# Load the configuration
with open("variables.yaml", "r", encoding="utf-8") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

MODEL_TYPE = config["MODEL_TYPE"]
MODEL_PATH = config["LOCAL_PATH"]["TEST_MODEL"]
UNIDATA_PATH = config["LOCAL_PATH"]["TEST_UNICLASS_DATASET"]
MULTIDATA_PATH = config["LOCAL_PATH"]["TEST_MULTICLASS_DATASET"]
COMPARE_MODELS_PATH = config["LOCAL_PATH"]["TEST_COMPARE_MODELS_DATASET"]
RESULTS_PATH = config["LOCAL_PATH"]["TEST_RESULTS"]

# Flags
ACCURACY = 0x01
CONFIDENCE = 0x02
CONFUSION_MATRIX = 0x04
TOP_CLASSES = 0x08
ALL = ACCURACY | CONFIDENCE | CONFUSION_MATRIX | TOP_CLASSES

def plot_and_save_mAccuracy_mConfidence(results, 
            save_path : str = RESULTS_PATH + "mAccuracy_mConfidence.png") -> None:
    """
    Create a table with the results of the mAccuracy and mConfidence for each model and save it
    
    :param results: the results of the mAccuracy and mConfidence
    :param save_path: the path to save the table
    """
    data = []

    for model, metrics in results.items():
        mean_accuracy = metrics["Mean accuracy"]
        classes_confidences = metrics["Confidence"]

        first_row = [model, "Mean confidence", classes_confidences["Mean confidence"], mean_accuracy]
        data.append(first_row)

        # Ajout des autres classes, mais en mettant '' pour éviter la répétition du modèle et de l'accuracy
        for cls, confidence in classes_confidences.items():
            if cls != "Mean confidence":  # On évite de le dupliquer
                data.append(["", cls, confidence, ""])

    df = pd.DataFrame(data, columns=["Model", "Class", "Confidence", "Mean Accuracy"])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    validate_path(save_path)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def calculate_mAccuracy_mConfidence(model_folder : str = MODEL_PATH,
                                    data_path : str = UNIDATA_PATH, 
                                    iteration : int = 10) -> dict :
    """
    Calculate the mAccuracy and mConfidence for each model in the model folder
    
    :param model_folder: the path to the model folder
    :param data_path: the path to the test data
    :param iteration: the number of iterations
    
    :return: the results of the mAccuracy and mConfidence for each model
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_files = [f for f in os.listdir(model_folder) if f.endswith(".pth")]
    nb_models = len(model_files)
    cpt = 1

    results = {}

    for model_file in model_files:
        model_path = os.path.join(model_folder, model_file)
        model_type = model_file.replace("model_", "").replace(".pth", "")
        model = load_model(model_type, model_path).to(device)
        
        dataloader, _, classes = get_dataloader(data_dir=data_path, batch_size=1, augment_data=False, test_size=0.0)

        # Initializations
        if model_type not in results:
            results[model_type] = {
                "Mean accuracy": 0,
                "Confidence": {"Mean confidence": 0}
            }
        
        for c in classes:
            results[model_type]["Confidence"][c] = 0

        # Calculations
        for it in range(iteration):
            print(f"{model_type} ({cpt}/{nb_models}), iteration {it+1}/{iteration}")
            accuracy = calculate_accuracy_uniclass(model, device, dataloader)
            results[model_type]["Mean accuracy"] += accuracy
            
            confidence, class_confidences = calculate_confidence_uniclass(model, device, dataloader)
            confidence = round(confidence * 100, 3)
            results[model_type]["Confidence"]["Mean confidence"] += confidence

            for i, class_confidence in enumerate(class_confidences):
                class_conf = round(class_confidence * 100, 3)
                results[model_type]["Confidence"][classes[i]] += class_conf

        # Mean calculations
        results[model_type]["Mean accuracy"] /= iteration
        results[model_type]["Confidence"]["Mean confidence"] /= iteration

        for c in classes:
            results[model_type]["Confidence"][c] /= iteration

        cpt += 1

    plot_and_save_mAccuracy_mConfidence(results)

    return results

def show_random_image_prediction_uniclass(model, device, dataloader : DataLoader,
                                          nb_images : int = 10, top : int = 3) :
    """
    Show the prediction for some random images from the dataset with the dataloader
    
    :param model: the model
    :param device: the device
    :param dataloader: the dataloader
    :param nb_images: the number of images to show
    :param top: the number of top classes to return
    """
    image_indexes = np.random.choice(len(dataloader), nb_images)
    
    for i in image_indexes:
        image, _ = dataloader.dataset[i]
        # to rgb
        image = image.unsqueeze(0).to(device)
        
        outputs = model(image)
        
        probabilities = torch.softmax(outputs, dim=1)
        top_prob, top_idx = probabilities.topk(top, dim=1)
        top_classes, top_probs = top_idx.squeeze(0).tolist(), top_prob.squeeze(0).tolist()
        
        top_classes = [config["CLASS_NAMES"][idx] for idx in top_classes]
        
        plt.imshow(image[0].permute(1, 2, 0).cpu())
        plt.axis('off')
        predictions_text = " | ".join([f"{classn}: {prob:.2%}" for classn, prob in zip(top_classes, top_probs)])
        plt.title(predictions_text)
        
        plt.show()

def show_random_image_prediction_multiclass(model, _device, 
                                            data_path : str = MULTIDATA_PATH, 
                                            nb_images : int = 10, top : int = 3) -> None:
    """
    Show the prediction for some random images from the dataset
    
    :param model: the model
    :param _device: the device (not used)
    :param data_path: the path to the dataset
    :param nb_images: the number of images to show
    :param top: the number of top classes to return
    """
    image_indexes = np.random.choice(len(os.listdir(data_path)), nb_images)
    
    for i in image_indexes:
        image_path = os.path.join(data_path, os.listdir(data_path)[i])        
        top_classes, top_probs = predict(model, image_path, top=top)
                
        image = Image.open(image_path) 
        plt.imshow(image)
        plt.axis('off')
        pred_text = " | ".join([f"{classn}: {prob:.2%}" for classn, prob in zip(top_classes, top_probs)])
        plt.title(pred_text)
        
        plt.show()

def test_uniclass(model_type : str = MODEL_TYPE,
                  model_path : str = MODEL_PATH,
                  data_path : str = UNIDATA_PATH, 
                  flags : int = ALL, top : int = 3, nb_images : int = 10) -> None:
    """
    Predict the test classes, print accuracy, confidese and confusion matrix and the top 
    classes and probabilities for each image
    
    :param model_type: the type of the model
    :param model_path: the path to the model
    :param data_path: the path to the test data
    :param flags: the flags to use
    :param top: the number of top classes to return
    """
    print(f"Testing model {model_type} with data from {data_path}")
    
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_type, model_path).to(device)

    # Load data
    dataloader, _, classes = get_dataloader(data_dir=data_path, batch_size=1, augment_data=False, test_size=0.0)
    
    # Calculate the accuracy
    if flags & ACCURACY:
        accuracy = calculate_accuracy_uniclass(model, device, dataloader)
        print(f"Accuracy: {accuracy:.2f}%")
    
    # Calculate the confidence
    if flags & CONFIDENCE:
        confidence, class_confidences = calculate_confidence_uniclass(model, device, dataloader)
        print (f"Confidence: {confidence*100:.2f}%")

        for i, class_confidence in enumerate(class_confidences):
            print(f"\t{classes[i]}: {class_confidence*100:.2f}%")
    
    # Calculate the confusion matrix
    if flags & CONFUSION_MATRIX:
        save_path = config["LOCAL_PATH"]["TEST_RESULTS"] + "/uniclass_test_confusion_matrix.png"
        validate_path(save_path)
        calculate_confusion_matrix_uniclass(model, device, dataloader, save=True, save_path=save_path)    

    # Show the top classes and probabilities
    if flags & TOP_CLASSES:
        show_random_image_prediction_uniclass(model, device, dataloader, top=top, nb_images=nb_images)

def test_multiclass(model_type : str = MODEL_TYPE,
                    model_path : str = MODEL_PATH,
                    data_path : str = MULTIDATA_PATH,
                    flags : int = ALL, top : int = 3, nb_images : int = 10) -> None:
    """
    Calculate the accuracy, confidence, confusion matrix and show the top classes and probabilities 
    for some images in the test data
    
    :param model_type: the type of the model
    :param model_path: the path to the model
    :param data_path: the path to the test data
    :param flags: the flags to use
    :param top: the number of top classes to return
    :param nb_images: the number of images
    """
    print(f"Testing model {model_type} with data from {data_path}")
    
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_type, model_path).to(device)
    
    # Create a manual dataset
    dataset = []
    with tqdm(total=len(os.listdir(data_path)), desc="Creating dataset") as pbar:
        for file in os.listdir(data_path):
            pbar.update(1)
            
            if file.endswith(".jpg") or file.endswith(".png"):
                classes, _, _ = file.split("_")
                accepted_classes = []
                for c in classes:
                    for char in config["CLASS_SYMBOLS"]:
                        if c == char:
                            accepted_classes.append(config["CLASS_NAMES"][config["CLASS_SYMBOLS"].index(char)])
                            break
    
                    if not c in config["CLASS_SYMBOLS"]:
                        print(f"Unknown class {c}")
                        continue
                
                dataset.append((os.path.join(data_path, file), accepted_classes))
    
    # Calculate the accuracy
    if flags & ACCURACY:
        accuracy = calculate_accuracy_multiclass(model, device, dataset)
        print(f"Accuracy: {accuracy:.2f}%")
    
    # Calculate the confidence
    if flags & CONFIDENCE:
        confidence, class_confidences = calculate_confidence_multiclass(model, device, dataset)
        print (f"Confidence: {confidence*100:.2f}%")
        classn = config["CLASS_NAMES"]
        for i, class_confidence in enumerate(class_confidences):
            print(f"\t{classn[i]}: {class_confidence*100:.2f}%")
    
    # Calculate the confusion matrix
    if flags & CONFUSION_MATRIX:
        save_path = config["LOCAL_PATH"]["TEST_RESULTS"] + "/multiclass_test_confusion_matrix.png"
        validate_path(save_path)
        calculate_confusion_matrix_multiclass(model, device, dataset, save=True, save_path=save_path)
    
    # Show the top classes and probabilities
    if flags & TOP_CLASSES:
        show_random_image_prediction_multiclass(model, device, data_path=data_path, nb_images=nb_images, top=top)
    
    return
    
def main(option : int = 0) -> None :
    """
    main function for the test module
    
    :param option: the option to choose (0 : test uniclass, 1 : test multiclass, 2 : test uniclass and multiclass, 3: compare models)
    """
    
    def verify_paths(paths : list) -> bool:
        """
        Verify if the path exists
        
        :param paths: the list of paths
        
        :return: True if all paths exist, False otherwise
        """
        
        err_paths = [path for path in paths if not os.path.exists(path)]
        if len(err_paths) > 0:
            print(f"Paths not found: {err_paths}")
            return False
        return True
    
    if option == 0: # Test uniclass
        print(f"Testing model {MODEL_TYPE} with data from {UNIDATA_PATH}")
        if not verify_paths([MODEL_PATH, UNIDATA_PATH]) : return
        test_uniclass(MODEL_TYPE, MODEL_PATH, UNIDATA_PATH, flags=ALL)

    elif option == 1: # Test multiclass        
        print(f"Testing model {MODEL_TYPE} with data from {MULTIDATA_PATH}")
        if not verify_paths([MODEL_PATH, MULTIDATA_PATH]) : return
        test_multiclass(MODEL_TYPE, MODEL_PATH, MULTIDATA_PATH, flags=ALL)
    
    elif option == 2: # Test uniclass and multiclass
        if not verify_paths([MODEL_PATH, UNIDATA_PATH, MULTIDATA_PATH]): return
        test_uniclass(MODEL_TYPE, MODEL_PATH, UNIDATA_PATH, flags=ALL)
        test_multiclass(MODEL_TYPE, MODEL_PATH, MULTIDATA_PATH, flags=ALL)
    
    elif option == 3: # Compare models
        if not verify_paths([COMPARE_MODELS_PATH, UNIDATA_PATH]): return
        calculate_mAccuracy_mConfidence(COMPARE_MODELS_PATH, UNIDATA_PATH, iteration=100)
    else:
        print("Unknown option")

if __name__ == "__main__":
    print("To train please got to ~/pfe/ and run \'python main.py test\'")
