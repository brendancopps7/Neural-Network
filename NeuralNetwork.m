%Clear everything and set up variables
clear;
rng(0);
raw = csvread("CTG.csv", 2, 0);
raw = raw(randperm(size(raw,1)),:);
n = .5; %Learning Parameter
M = 20; %number of hidden layer nodes
K = size(unique(raw(:, size(raw, 2))),1); %number of output nodes
%Set up training and testing data
rawtrain = raw(1:ceil((2/3)*size(raw, 1)), :);
rawtest = raw(ceil((2/3)*size(raw,1))+1:size(raw,1),:);
means = mean(rawtrain);
stds = std(rawtrain);
train = [];
test = [];
%Standardize Data
for i = 1:size(raw, 2)-1
    tempcol = (rawtrain(:, i)-means(i))./stds(i);
	train = [train tempcol];
	tempcol = (rawtest(:, i)-means(i))./stds(i);
	test = [test tempcol];
end
%Make Model and Prediction
Y = zeros(size(train, 1), K);
for i =1:size(train, 1)
    Y(i, rawtrain(i, size(rawtrain, 2))) = 1;
end
inputvector = [ones(size(train, 1), 1), train];
beta = (2*rand(size(train, 2)+1, M))-1;
theta = (2*rand(M, K))-1;
trainingacc = ones(1000,1);
for i = 1:1000
    H = 1./(1+exp(-(inputvector*beta)));
    O = 1./(1+exp(-(H*theta)));
    deltaout = Y - O;
    theta = theta + (n/size(train, 1))*transpose(H)*deltaout;
    deltahidden = deltaout*transpose(theta).*H.*(1-H);
    beta = beta + (n/size(train, 1))*transpose(inputvector)*deltahidden;
    currentacc = 0;
    
    for j = 1:size(train, 1)
        if(rawtrain(j, size(rawtrain, 2)) == round(find(O(j,:)==max(O(j,:)))))
            currentacc = currentacc + 1;
        end 
    end
    trainingacc(i) = currentacc;
end
trainingacc = trainingacc./size(train,1);
plot(1:1000, trainingacc)
%Test Data
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
