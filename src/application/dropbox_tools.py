import dropbox

def is_image_file(filename) -> bool:
    """
    Check if the filename has a correct extension (.jpg, .jpeg, .png, .bmp).
    """
    return filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png') or filename.endswith('.bmp')

def rename_images_on_dropbox(dbx, folder_name):
    """
    Rename the images in the specified folder on Dropbox.
    """
    try:
        # List files on Dropbox in the specified folder
        result = dbx.files_list_folder(folder_name)
        
        # Count the number of .jpg images and rename them if necessary
        cpt = 0
        classn = folder_name.rstrip('/').split('/')[-1]
        for entry in result.entries:
            if isinstance(entry, dropbox.files.FileMetadata) and is_image_file(entry.name):
                cpt += 1
                
                # optimize
                end = entry.name.split('_')[-1]
                number = end.split('.')[0]
                ext = end.split('.')[1]
                
                if number != str(cpt):
                    # rename the file
                    print(f"Renaming {entry.name} to {classn}_{cpt}.{ext}")
                    dbx.files_move_v2(entry.path_lower, f"{folder_name}/{classn}_{cpt}.{ext}")

        return cpt
    except dropbox.exceptions.ApiError as err:
        #print(f"Folder {folder_name} not found on Dropbox")
        return 0

def count_images_on_dropbox(dbx, folder_name):
    """
    Count the number of .jpg images already uploaded in the specified folder on Dropbox.
    """
    rename_images_on_dropbox(dbx, folder_name)
    
    try : 
        result = dbx.files_list_folder(folder_name)
        
        # Count the number of .jpg images
        cpt = 0
        for entry in result.entries:
            if isinstance(entry, dropbox.files.FileMetadata) and is_image_file(entry.name):
                cpt += 1
                
        return cpt
    except dropbox.exceptions.ApiError as err:
        #print(f"Folder {folder_name} not found on Dropbox")
        return 0
    
def import_images_from_dropbox(dbx, remote_folder, local_folder):
    """
    Import images from the specified folder on Dropbox to the local folder.
    
    :param dbx: Dropbox object
    :param remote_folder: the path of the folder on Dropbox
    :param local_folder: the path of the local folder
    """
    rename_images_on_dropbox(dbx, remote_folder)
    
    try : 
        result = dbx.files_list_folder(remote_folder)
        
        # Count the number of .jpg images and rename them if necessary
        for entry in result.entries:
            if isinstance(entry, dropbox.files.FileMetadata) and is_image_file(entry.name):                
                print(f"Donwloading image {entry.name} to {local_folder}/{entry.name}")
                dbx.files_download_to_file(f"{local_folder}/{entry.name}", f"{remote_folder}/{entry.name}")
        
    except dropbox.exceptions.ApiError as err:
        #print(f"Folder {remote_folder} not found on Dropbox")
        return 0
