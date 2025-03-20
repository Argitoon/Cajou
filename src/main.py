from typing import List, Tuple, Literal 
import classifier.dataloader as dataloader
import classifier.train as train
import classifier.tools as tools
import classifier.model as model
import application.appli as appli
import tests.test  as test
import sys

def execute(name : str = "application") -> None :
    """
    Execute specified code in the specific module (or part of a module)
    
    :param name: the name of the module (or part of) to execute
    """
    # Application module
    if name == "application" or name == "appli":
        print("Starting application")
        appli.main()
    
    # Train module
    elif name == "train":
        print("Starting training")
        train.main(0) # 0 = default
    elif name == "apptrain":
        print("Starting application training")
        train.main(1) # 1 = application
        
    # Test module
    elif name == "test":
        print("Starting testing (uniclass and multiclass)")
        test.main(2) # 2 = uniclass and multiclass test 
    elif name == "unitest":
        print("Starting testing (uniclass)")
        test.main(0) # 0 = uniclass test
    elif name == "multitest":
        print("Starting testing (multiclass)")
        test.main(1) # 1 = multiclass test
    elif name == "modeltest":
        print("Starting testing (compare models)")
        test.main(3) # 3 = compare models test
    
    # Model module
    elif name == "model":
        print("Starting model")
        model.main()
    
    # Tools module
    elif name == "tools":
        print("Starting tools")
        tools.main()
    
    # Dataloader module
    elif name == "dataloader":
        print("Starting dataloader")
        dataloader.main()
    
    # Help
    elif name == "help":
        print("List of modules: application, train, apptrain, test, unitest, multitest, modeltest, model, tools, dataloader")   
    else:
        print("Unknown module")
        
if __name__ == "__main__":
    if len(sys.argv) > 1:
        execute(sys.argv[1])
    else:
        execute()
        
    
    
    
        
    
    