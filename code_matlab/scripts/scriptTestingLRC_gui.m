clear all;
close all;
clc;
addpath('../');


global idDigit;


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


%-- Do a first prediction
rp = randperm(m);
idDigit = 1;
ind = rp(idDigit);
img = reshape(X(ind,2:end),[heightDigit widthDigit]);
pred = lrc.predict(all_theta, X(ind,:));
ref = y(ind);


%-- Display result
h = figure; imagesc(img); axis image; colormap(gray); title('Press left/right to navigate / middle to quit'); axis off;
if (pred == ref)
    text(2,3,{['True value = ',num2str(ref)],['Prediction = ',num2str(pred)]},'Color','green','FontSize',10);
else
    text(2,3,{['True value = ',num2str(ref)],['Prediction = ',num2str(pred)]},'Color','red','FontSize',10);
end


%-- Set a callback function with the 'WindowButtonDownFcn' event for the rest of the application
set(gcf,'WindowButtonDownFcn', {@visu.clickDownGui,h,X,y,all_theta,widthDigit,heightDigit,rp});
