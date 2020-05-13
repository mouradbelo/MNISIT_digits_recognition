function [p,temp] = predict(all_theta, X)

    %-- PREDICT Predict the label for a trained multiple logistic regression classifiers. The labels 
    %-- are in the range 1..K, where K = size(all_theta, 1). 
    %-- p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
    %-- for each example in the matrix X. Note that X contains the examples in
    %-- rows. all_theta is a matrix where the i-th row is a trained logistic
    %-- regression theta vector for the i-th class. 

%    m = size(X, 1);
%    X = [ones(m, 1) X]; %-- Add ones to the X data matrix        
    temp = lrc.sigmoid(X*all_theta');
    [~,p] = max(temp,[],2);
    p = p-1;            %-- the first element of the array corresponds to digit 0

end
