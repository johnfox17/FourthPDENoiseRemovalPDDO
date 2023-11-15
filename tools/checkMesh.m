clear all;
close all;
addpath('../data/');
mesh = table2array(readtable('mesh.csv'));
figure; plot(mesh(:,1),mesh(:,2),'o')