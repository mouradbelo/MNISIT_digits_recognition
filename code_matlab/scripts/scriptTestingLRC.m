clear all;
close all;
clc;
addpath('../');


%-- mnist database location
url = 'https://www.creatis.insa-lyon.fr/~bernard/ge/';
local_data_path = '../data/';
local_param_path = '../param/';


%-- Downlad minst database
filename_db = 'mnist.mat';
if (~exist([local_data_path,filename_db],'file'))
     tools.download(filename_db, url, local_data_path);
end


%-- Load mnist database
load([local_data_path,filename_db]);
widthDigit = size(test.images,2);
heightDigit = size(test.images,1);


%-- Load parameters
%-- Save learned parameters
filename_param = 'param_mnist.mat';
load([local_param_path,filename_param]);


%-- Create X matrix
X = zeros(size(test.images,3),widthDigit*heightDigit+1);
for k=1:size(test.images,3)
    digit = test.images(:,:,k);
    X(k,:) = [1,digit(:)'];
end
y = test.labels;
m = size(X,1);


%-- Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);
visu.displayDatabase(sel,widthDigit,heightDigit);


%-- Evaluate the performance of the learned method from the full testing database
%-- Predict for One-Vs-All
pred = lrc.predict(all_theta, X);
disp(['Testing Set Accuracy: ',num2str(mean(double(pred == y)) * 100)])

