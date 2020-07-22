close all; clear all; clc;
color1 =[50,255,0];
color2 =[255,0,0];
% color2 =[0,250,0];
for i =0:209
    str1 = ['C:\Users\zjh\Desktop\imagesforexperiment\26\patchwithbuffer\',int2str(i),'.png'];
    str2 = ['I:\imagesforexperiment\26\test\GTM\',int2str(i),'.png'];
    str3 = ['C:\Users\zjh\Desktop\imagesforexperiment\26\densenoholepostprocessing\original_g_binception4_13_layersame\1\',int2str(i),'.png'];
    
    patch = imread(str1);
    img = imread(str2);
    pixel = imread(str3);
    pixel = imresize(pixel,[256,256]);
    pixel = im2bw(pixel);
    img1 = imresize(img,[256,256]);
    ptach1 = double(patch);
    pixel1 = double(pixel);
    
    
    mat = ones(256,256)*color1([1]);
    mat1 = ones(256,256)*color1([2]);
    mat2 = ones(256,256)*color1([3]);
    mat3 = cat(3,mat,mat1,mat2);
    mat3 = mat3.*ptach1;
    mat4 = uint8(mat3);
    
    mat = ones(256,256)*color2([1]);
    mat1 = ones(256,256)*color2([2]);
    mat2 = ones(256,256)*color2([3]);
    mat3 = cat(3,mat,mat1,mat2);
    mat3 = mat3.*pixel1;
    mat5 = uint8(mat3);
    
    K = imlincomb(1,img1,1,mat5,1,mat4);
%     K = imlincomb(1,img1,1,mat4);
%     figure,imshow(K);
    imwrite(img1,['I:\imagesforexperiment\26\test\resize_GT\',int2str(i),'.png']);
end