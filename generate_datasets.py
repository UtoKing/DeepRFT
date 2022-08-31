import os
import random
import shutil

data_path = r"./Datasets/GoPro"
target_path = r"./Datasets/GoPro_Small"

picture_list1=os.listdir(os.path.join(data_path,"train","blur"))
picture_list2=os.listdir(os.path.join(data_path,"test","blur"))
random.shuffle(picture_list1)
random.shuffle(picture_list2)

for i in range(1500):
    shutil.copyfile(os.path.join(data_path,"train","blur",picture_list1[i]),os.path.join(target_path,"train","blur",picture_list1[i]))
    shutil.copyfile(os.path.join(data_path,"train","sharp",picture_list1[i]),os.path.join(target_path,"train","sharp",picture_list1[i]))

    
    
for i in range(500):
    shutil.copyfile(os.path.join(data_path,"test","sharp",picture_list2[i]),os.path.join(target_path,"test","sharp",picture_list2[i]))
    shutil.copyfile(os.path.join(data_path,"test","blur",picture_list2[i]),os.path.join(target_path,"test","blur",picture_list2[i]))