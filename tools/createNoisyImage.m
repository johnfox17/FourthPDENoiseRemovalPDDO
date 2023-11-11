clear all;
close all;
addpath('../data/');
%read image
lena  = imread('Lena.png');
%convert to grayscale
lena = rgb2gray(lena);
%create gaussian noise
noisyLena = imnoise(lena,'gaussian');
figure; imagesc(lena);
colormap gray
figure; imagesc(noisyLena)
colormap gray
imwrite(noisyLena,'../data/noisyLena.png')
imwrite(lena,'../data/lenaGrayScale.png')
