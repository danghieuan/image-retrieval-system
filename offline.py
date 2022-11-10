import os   # interacting with the operating system
import numpy as np  # manipulating array
from tensorflow.keras.preprocessing import image as kimage   # read image, resize image               
from feature_extractor import FeatureExtractor

# Define root path for image database and feature database
root_image_path = "./static/image_database/"
root_feature_path = "./static/feature_database/"


# Path of image folder
def folder_to_images(folder):

    list_dir = [folder + "/" + name for name in os.listdir(folder) if name.endswith((".jpeg", ".jpg", ".png"))] # Only accept .jpeg, .jpg, .png images
    
    k = 0
    
    images_np = np.zeros(shape=(len(list_dir), 224, 224, 3))

    images_path = []
   
    for path in list_dir:
        try:
            img = kimage.load_img(path, target_size = (224, 224))   # resize image to 224x224x3
            images_np[k] = kimage.img_to_array(img, dtype = np.float32) # change uint8 => float32 [0, 1] for easy caculating
            images_path.append(path)
            k += 1
            
        except Exception:
            print("error: ", path)          # print path error

    images_path = np.array(images_path)
    return images_np, images_path           # return path of image and array of that image



if __name__ == "__main__":

    fe = FeatureExtractor()
    
    for folder in os.listdir(root_image_path):
        path = root_image_path + folder
        # print(path)
        images_np, images_path = folder_to_images(path)
        np.savez_compressed(root_feature_path + folder, array_1 = np.array(images_path), array_2 = fe.extract(images_np)) # 


imgs_feature = []   # feature list
paths_feature = []  # path list


for folder in os.listdir(root_image_path):  # read all folder in folder list (image database)
    path = root_image_path + folder          
    print(path)
    images_np, images_path = folder_to_images(path)     # export numpy of image and path image
    paths_feature.extend(np.array(images_path))         # Gộp các feature vào feature list
    imgs_feature.extend(fe.extract(images_np))          

# Save feature        
np.savez_compressed(root_feature_path + "concat_all_feature", array_1 = np.array(paths_feature), array_2 = np.array(imgs_feature))