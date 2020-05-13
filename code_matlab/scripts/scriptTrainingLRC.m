clear all;
close all;
clc;
addpath('../');


%-- parameters
maxIter = 10;   %-- gradient descent: maximum number of iterations
epsilon = 0.01; %-- gradient descent: convergence thresholder
tau = 1;        %-- gradient descent: learning rate coefficient 
        

%-- mnist database location
url = 'https://www.creatis.insa-lyon.fr/~bernard/ge/';
local_data_path = '../data/';
local_param_path = '../param/';


%-- Downlad minst database
filename_db = 'mnist.mat';
if (~exist([local_data_path,filename_db],'file'))
     tools.download(filename_db,url,local_data_path);
end


%-- Load mnist database
load([local_data_path,filename_db]);
widthDigit = size(training.images,2);
heightDigit = size(training.images,1);


%-- Perform training
num_labels = 10;          %-- 10 labels, from 0 to 9


%-- Create X matrix
X = zeros(size(training.images,3),widthDigit*heightDigit+1);
for k=1:size(training.images,3)
    digit = training.images(:,:,k);
    X(k,:) = [1,digit(:)'];
end


%-- Create y vector
y = training.labels;
m = size(X,1);


%-- Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);
visu.displayDatabase(sel,widthDigit,heightDigit);


%--  train logistic regression method
disp('\nTraining Logistic Regression...\n')
[all_theta] = lrc.train(X, y, num_labels, maxIter, epsilon, tau);


%-- Save learned parameters
filename_param = 'param_mnist.mat';
if (~exist(local_param_path,'dir'))
     mkdir(local_param_path);
end
save([local_param_path,filename_param],'all_theta');
disp(['Parameters saved to ',[local_param_path,filename_param]])


%-- Predict for One-Vs-All on the training dataset
pred = lrc.predict(all_theta, X);
disp(['Training Set Accuracy: ',num2str(mean(double(pred == y)) * 100)])

