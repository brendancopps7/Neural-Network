%More information about this code can be found in the associated README file
%Additional code samples can be found on my github: https://github.com/brendancopps7

%Clear everything and set up variables to ensure clean workspace
clear;
rng(0);
raw = csvread("CTG.csv", 2, 0);
%Data for this assignment was taken from a file called CTG.csv,
%this can be substituted for any file as long as "raw" only contains data (no headers or labels)
raw = raw(randperm(size(raw,1)),:);
%A requirement of the class was to randomize rows. This is not necesary for the code to function
n = .5; %Learning Parameter
%The learning parameter deals with how quickly the code will move along the gradient
%A low number would lead to slow convergence
%A high number runs the risk of constantly overshooting
M = 20; %number of hidden layer nodes
%This code only involves one hidden layer, so the number of nodes is a scalar
K = size(unique(raw(:, size(raw, 2))),1); %number of output nodes
%This can be changed manually, but works under the assumption that the last column of data holds the labels for testing
%Furthermore, it assumes that all labels are present in the final column and that all labels in the final column will be used
N = 1000;
% 1000 is an arbitrarily large number and can be increased/decreased depending on resources/constraints

%Set up training and testing data
%For this code, I chose to use 2/3 of the data as training and use 1/3 to test the accuracy of my code
rawtrain = raw(1:ceil((2/3)*size(raw, 1)), :);
rawtest = raw(ceil((2/3)*size(raw,1))+1:size(raw,1),:);
means = mean(rawtrain);
stds = std(rawtrain);
train = [];
test = [];
%Standardize Data
for i = 1:size(raw, 2)-1 % -1 because labels are not standardized and remain in the raw dataset
    tempcol = (rawtrain(:, i)-means(i))./stds(i);
	train = [train tempcol];
end
%hold onto these means and stds because we will use them to standardize the testing data as well

%Make Model and Prediction
Y = zeros(size(train, 1), K);
for i =1:size(train, 1)
    Y(i, rawtrain(i, size(rawtrain, 2))) = 1;
end
%This loop converts a single column of labels into K columns of binary labels
%Ex: This code would turn:
%
%[1     into    [1 0 0
% 2              0 1 0
% 3]             0 0 1]
%

inputvector = [ones(size(train, 1), 1), train];
beta = (2*rand(size(train, 2)+1, M))-1;
theta = (2*rand(M, K))-1;
trainingacc = ones(1000,1);
for i = 1:N
    H = 1./(1+exp(-(inputvector*beta)));
    O = 1./(1+exp(-(H*theta)));
    deltaout = Y - O;
    theta = theta + (n/size(train, 1))*transpose(H)*deltaout;
    deltahidden = deltaout*transpose(theta).*H.*(1-H);
    beta = beta + (n/size(train, 1))*transpose(inputvector)*deltahidden;
    currentacc = 0;

    %Just for graphing purposes, we calculate the accuracy at each step and store them
    for j = 1:size(train, 1)
        if(rawtrain(j, size(rawtrain, 2)) == round(find(O(j,:)==max(O(j,:)))))
            currentacc = currentacc + 1;
        end 
    end
    trainingacc(i) = currentacc;
end
trainingacc = trainingacc./size(train,1);
%The next line, if uncommented, shows how the accuracy of the training data grows with the size of N
%plot(1:N, trainingacc)



%Test Data
for i = 1:size(raw, 2)-1 % -1 because labels are not standardized and remain in the raw dataset
    tempcol = (rawtest(:, i)-means(i))./stds(i);
    test = [test tempcol];
end
inputvector = [ones(size(test, 1), 1), test];
H = 1./(1+exp(-(inputvector*beta)));
O = 1./(1+exp(-(H*theta)));
testacc = 0;
for j = 1:size(test, 1)
    if(rawtest(j, size(rawtest, 2)) == round(find(O(j,:)==max(O(j,:)))))
        testacc = testacc + 1;
    end
end
testacc = testacc./size(test, 1);
%This last variable "testacc" describes the accuracy of the Neural Network on the tested data.
