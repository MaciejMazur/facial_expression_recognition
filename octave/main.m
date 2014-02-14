

%  0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral


load('../data/All.mat');
% Change rage from 0-6 to 1-7
A(:,1) = A(:,1) .+ 1;
A(:,2:end-1) = normalize(A(:,2:end-1));
[m,n] = size(A);
A = A(randperm(size(A,1)),:);
d1 = 48;
d2 = 48;

train = A(A(:,end) == 1, 1:end-1);
trainY = train(:,1);
trainX = train(:,2:end);

public = A(A(:,end) == 2, 1:end-1);
publicY = public(:,1);
publicX = public(:,2:end);

private = A(A(:,end) == 3, 1:end-1);
privateY = private(:,1);
privateX = private(:,2:end);

layers = [d1*d2 50 50 7];
%15 sec - iteration
%layers = [d1*d2 100 100 100 100 100 7];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% gradient checking
%%%%%%%%%%%%%%%%%%%%%%%%
%layers = [d1*d2 3 3 7];

%params = initializeParams(layers);
%labels = 7;
%lambda = 0;
%[a b] = costFunction(params, layers,trainX(1:100,:), trainY(1:100,:), lambda, labels);

%costFunc = @(p) costFunction(p, layers, trainX(1:100,:), trainY(1:100,:), lambda, labels)
%numgrad = numGrad(costFunc, params);
%diff = norm(numgrad-grad)/norm(numgrad+grad)
labels = 7;
delta = 0.1;
lambda = 0.01;
iterations = 10;

t = 22900;
%layers = [d1*d2 1000 700 500 250 150 50 7];
layers = [d1*d2 100 100 100 100 100 7];
params = initializeParams(layers);
options = optimset('MaxIter', 1000);

costF = @(p) costFunction(p, layers, trainX(1:t,:), trainY(1:t,:), lambda, labels)
[nn_params, cost] = fmincg(costF, params , options);
pred = predict(nn_params,layers,trainX);
errorRate(pred, trainY)

pred = predict(nn_params,layers,trainX(1:t,:));
errorRate(pred, trainY(1:t,:))

pred = predict(nn_params,layers,trainX(t+1:end,:));
errorRate(pred, trainY(t+1:end,:))
%layers = [d1*d2 1000 500 250 100 50 7];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% stochastic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%layers = [d1*d2 100 100 100 100 100 7];
%params = initializeParams(layers);
%labels = 7;
%delta = 0.1;
%lambda = 0;
%iterations = 10;
%[sg_params cost] = stochasticGradient(params, layers, trainX, trainY, lambda, labels, delta, iterations); 
%%pred = predict(nn_params,layers,trainX);
%errorRate(pred, trainY)


%options = optimset('MaxIter', 100);
%costF = @(p) costFunction(p, layers, trainX, trainY, lambda,labels);
%[nn_params, cost] = fmincg(costF, params , options);
%pred = predict(nn_params,layers,trainX);
%errorRate(pred, trainYr
