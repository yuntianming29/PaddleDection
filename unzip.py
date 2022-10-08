import os
import zipfile
 
f = zipfile.ZipFile("/home/aistudio/data/data126280/HelmetDetection.zip",'r') # 压缩文件位置
for file in f.namelist():
    f.extract(file,"/home/aistudio/data/data126280/")               # 解压位置
f.close()