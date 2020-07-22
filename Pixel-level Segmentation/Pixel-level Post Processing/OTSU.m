close all,clear all,clc ;
dir=('C:\Users\zjh\Desktop\论文代码与数据\imagesforexperiment\26\test\predict\original_g\3');
% img_dir1 = dir([dir,'\*.png']);
trainingSet = imageSet(dir,'recursive');
n = numel(trainingSet);
numImages = trainingSet(n).Count;
for i =0:(numImages-1)
    str = ['C:\Users\zjh\Desktop\论文代码与数据\imagesforexperiment\26\test\predict\original_g\3\',int2str(i),'_predict.png'];
    original_img = imread(str);
    img = im2double(original_img);
    T = graythresh(img);
    J = im2bw(img,T);
%     figure,imshow(J);
    imwrite(J,['C:\Users\zjh\Desktop\论文代码与数据\imagesforexperiment\26\Otsu_for_netresults\original_g\3\',int2str(i),'.png']); 
end