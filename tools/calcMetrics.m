addpath('../data')
lena = single(imread('lena.png'));
denoisedLenaPaper = table2array(readtable('denoisedLena_Paper.csv'));
denoisedLenaPDDO = table2array(readtable('denoisedLena_PDDO.csv'));

%filter paper method for huge values
denoisedLenaPaper = denoisedLenaPaper(2:513,2:513);
denoisedLenaPaper(denoisedLenaPaper>255)=255;
denoisedLenaPaper(denoisedLenaPaper<-255)=-255;

figure; imagesc(lena); colormap gray;
figure; imagesc(denoisedLenaPaper); colormap gray;
figure; imagesc(denoisedLenaPDDO); colormap gray;

%Calculate PSNR
paperPSNR = 10.*log10(255^2/(norm(lena(:)-denoisedLenaPaper(:)))^2);
PDDOPSNR = 10.*log10(255^2/(norm(lena(:)-denoisedLenaPDDO(:)))^2);

%Calculate SSIM
lenaMean = mean(lena(:));
paperMean = mean(denoisedLenaPaper(:));
pddoMean = mean(denoisedLenaPDDO(:));
lenaVar = var(lena(:));
paperVar = var(denoisedLenaPaper(:));
pddoVar = var(denoisedLenaPDDO(:));
lenaPaperCovar = xcov(lena(:),denoisedLenaPaper(:));
paperSSIM = (2*lenaMean*paperMean+ 1e-9);