close all;
clear;
clc;
%Cut the image into 100 pieces
% tr_dir=uigetdir('select the data sets...');
tr_dir1='D:\U5510\Random1\26_cut\val\GTM';
tr_dir2='D:\U5510\Random1\26_cut\val\original_g';
% [priPath,dset]=priorPath(tr_dir);
img_dir1 = dir([tr_dir1,'\*.png']);
img_dir2 = dir([tr_dir2,'\*.png']);
img_num = length(img_dir1);
for n = 1:1:img_num
  fname=img_dir1(n).name;
  imgdata1=imread([tr_dir1,'\',fname]);
  imgdata2=imread([tr_dir2,'\',fname]);
  [L,W,N] = size(imgdata1);
  counter = 0;
  for i= 1:1:L
      for j= 1:1:W
          if(imgdata1(i,j)>0)
              counter=counter+1;
          end
      end
  end
  if (counter>3072)
%     for i= 1:1:L
%       for j= 1:1:W
%           imgdata(i,j)=255;
%               
%       end
%     end
   imwrite(imgdata2,['D:\U5510\Random1\26_cut\0.75\val\YES\',fname]);
  else
%      for i= 1:1:L
%       for j= 1:1:W
%           imgdata(i,j)=0;
%               
%       end
%      end
   imwrite(imgdata2,['D:\U5510\Random1\26_cut\0.75\val\NO\',fname]);  
  end
end