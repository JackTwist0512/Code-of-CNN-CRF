close all; clear all; clc;

for j  = 2:2:40
    mkdir(['C:\Users\zjh\Desktop\Unet����ļ�\����\CRF\test\26\save_denseCRF\post_process\���ؼ������ͺ���\0_���ؼ������ͺ���_',int2str(j),'pixel']);
    mkdir(['C:\Users\zjh\Desktop\Unet����ļ�\����\CRF\test\26\save_denseCRF\post_process\patch���𻺳������\0_patch���𻺳������_',int2str(j),'pixel']);
    for i =0:209
        str1 = ['I:\imagesforexperiment\26\densenoholepostprocessing\original_g_binception4_13_layersame\1\',int2str(i),'.png'];%���ؼ���ָ���
        str2 = ['D:\U5510\Random1\VGG16_26\0.5\seg_result_bw0.5\',int2str(i),'.png'];%patch����ָ���
        img = imread(str1);
        patch_img = imread(str2);
        patch_img = im2bw(patch_img);
        patch_img1 = imresize(patch_img,[256,256]);

        bw= img;
        se = strel('square',3);
        for z =1:1:j
            bw = imdilate(bw,se);
        end
    %     img1 = xor(img,bw);
        imwrite(bw,['C:\Users\zjh\Desktop\Unet����ļ�\����\CRF\test\26\save_denseCRF\post_process\���ؼ������ͺ���\0_���ؼ������ͺ���_',int2str(j),'pixel\',int2str(i),'.png']); 
        patch_cut_img = bw&patch_img1;
        imwrite(patch_cut_img,['C:\Users\zjh\Desktop\Unet����ļ�\����\CRF\test\26\save_denseCRF\post_process\patch���𻺳������\0_patch���𻺳������_',int2str(j),'pixel\',int2str(i),'.png']); 
    %     figure,imshow(patch_cut_img);
    end
end