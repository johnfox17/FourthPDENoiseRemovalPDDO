clear all;
close all;
addpath('C:\Users\docta\Documents\Thesis\FourthPDENoiseRemovalPDDO\data');
%denoisedLena = str2double(table2array(readtable('deNoisedImages.csv')));
denoisedLena = table2array(readtable('deNoisedImages.csv'));
image = denoisedLena(1,:);
image(image>255)=255;
image(image<-255)=-255;
figure; imagesc(reshape(image,[512 512]).'); colormap gray;
title('Time Step - 1')
figure; surf(reshape(image,[512 512]).')
title('Time Step - 1')

image = denoisedLena(50,:);
image(image>255)=255;
image(image<-255)=-255;
figure; imagesc(reshape(image,[512 512]).'); colormap gray;
title('Time Step - 50')
figure; surf(reshape(image,[512 512]).')
title('Time Step - 50')

image = denoisedLena(100,:);
image(image>255)=255;
image(image<-255)=-255;
figure; imagesc(reshape(image,[512 512]).'); colormap gray;
title('Time Step - 100')
figure; surf(reshape(image,[512 512]).')
title('Time Step - 100')

image = denoisedLena(150,:);
image(image>255)=255;
image(image<-255)=-255;
figure; imagesc(reshape(image,[512 512]).'); colormap gray;
title('Time Step - 150')
figure; surf(reshape(image,[512 512]).')
title('Time Step - 150')

image = denoisedLena(200,:);
image(image>255)=255;
image(image<-255)=-255;
figure; imagesc(reshape(image,[512 512]).'); colormap gray;
title('Time Step - 200')
figure; surf(reshape(image,[512 512]).')
title('Time Step - 200')

image = denoisedLena(500,:);
image(image>255)=255;
image(image<-255)=-255;
figure; imagesc(reshape(image,[512 512]).'); colormap gray;
title('Time Step - 500')
figure; surf(reshape(image,[512 512]).')
title('Time Step - 500')

image = denoisedLena(800,:);
image(image>255)=255;
image(image<-255)=-255;
figure; imagesc(reshape(image,[512 512]).'); colormap gray;
figure; surf(reshape(image,[512 512]).')
title('Time Step - 800')

image = denoisedLena(1000,:);
image(image>255)=255;
image(image<-255)=-255;
figure; imagesc(reshape(image,[512 512]).'); colormap gray;
figure; surf(reshape(image,[512 512]).')

image = denoisedLena(400,:);
idx = image>255;
image(idx)=200;
idx = image<-255;
image(idx)=-200;

figure; imagesc(reshape(image,[512 512]).'); colormap gray;
figure; surf(reshape(image,[512 512]).')

image = denoisedLena(500,:);
idx = image>255;
image(idx)=200;
idx = image<-255;
image(idx)=-200;

figure; imagesc(reshape(image,[512 512]).'); colormap gray;
figure; surf(reshape(image,[512 512]).')