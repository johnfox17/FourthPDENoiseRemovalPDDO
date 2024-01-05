close all;
clear all;
addpath('../data')
lena = single(imread('lena.png'));
denoisedLenaPaper = table2array(readtable('denoisedLena_Paper.csv'));
denoisedLenaPDDO = table2array(readtable('denoisedLena_PDDO.csv'));

%filter paper method for huge values
denoisedLenaPaper(denoisedLenaPaper>255)=255;
denoisedLenaPaper(denoisedLenaPaper<-255)=-255;

figure; imagesc(lena); colormap gray;
figure; imagesc(denoisedLenaPaper); colormap gray;
figure; imagesc(denoisedLenaPDDO); colormap gray;

%Calculate PSNR
paperPSNR = 10.*log10(255^2/(norm(lena(:)-denoisedLenaPaper(:)))^2);
pddoPSNR = 10.*log10(255^2/(norm(lena(:)-denoisedLenaPDDO(:)))^2);

%Calculate SSIM
[mssim_Paper, ssim_map_Paper] = ssim_index(lena, denoisedLenaPaper);
[mssim_PDDO ssim_map_PDDO] = ssim_index(lena, denoisedLenaPDDO);