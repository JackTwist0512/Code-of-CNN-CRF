import cv2 as cv
import numpy as np
import os
#创建文件夹
def makedir(path):

    folder = os.path.exists(path)

    if folder is True:
        print("文件夹已存在")
    else:
        os.makedirs(path)
        print("文件夹创建中")
        print("完成")

def cut_img(img,save_path,cols,rows,n):
    img = img
    sum_rows = img.shape[0]  # 高度
    sum_cols = img.shape[1]  # 宽度
    for i in range(int(sum_cols/cols)):
        for j in range(int(sum_rows/rows)):
            cut_img = img[j*rows:(j+1)*rows,i*cols:(i+1)*cols,:]
            img_save_path = save_path + "/"+str(n)+"c"+str(i)+"_r"+str(j)+".png"
            cv.imwrite(img_save_path,cut_img)


# img = cv.imread("C:/Users/zjh/Desktop/0.jpg")
# save_path = "D:/cut/0"
# makedir(save_path)
for n in range(27,29):
    for j in range(3):
        list = ("test","train","val")
        name= list[j]
        img_path ="D:/U5510/Random1/"+str(n)+"_resize/"+name+"/original"
        img_g_path = "D:/U5510/Random1/"+str(n)+"_resize/"+name+"/original_g"
        gt_path ="D:/U5510/Random1/"+str(n)+"_resize/"+name+"/GTM"
        img_save_path ="D:/U5510/Random1/"+str(n)+"_cut/"+name+"/original"
        img_g_save_path = "D:/U5510/Random1/"+str(n)+"_cut/"+name+"/original_g"
        gt_save_path ="D:/U5510/Random1/"+str(n)+"_cut/"+name+"/GTM"
        makedir(img_save_path)
        makedir(img_g_save_path)
        makedir(gt_save_path)
        img_list = os.listdir(img_path)
        img_g_list = os.listdir(img_g_path)
        gt_list = os.listdir(gt_path)
        num1 = len(img_list)
        num2 = len(img_g_list)
        num3 = len(gt_list)
        if num1 == num2 == num3:
            for i in range(num1):
                img = cv.imread(img_path+"/"+str(i)+"_resize.png")
                img_g = cv.imread(img_g_path+"/"+str(i)+"_resize.png")
                gt = cv.imread(gt_path+"/"+str(i)+"_resize.png")
                cut_img(img,img_save_path,64,64,i)
                cut_img(img_g, img_g_save_path, 64, 64, i)
                cut_img(gt, gt_save_path, 64, 64, i)

# cut_img(img,save_path,100,100)
# merge_img("D:/cut/0","D:/cut/0/0.png",10,10)