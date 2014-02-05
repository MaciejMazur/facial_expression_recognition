

%  0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral


load('../data/All.mat');
% Change rage from 0-6 to 1-7
A(:,1) = A(:,1) .+ 1;
A(:,2:end-1) = normalize(A(:,2:end-1));
[m,n] = size(A);

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

layers = [d1*d2 100 7];
params = initializeParams(layers);
[a b] = costFunction(params, layers,trainX, trainY, 0);
lambda = 0;
options = optimset('MaxIter', 10);
costF = @(p) costFunction(p, layers, trainX, trainY, lambda);
[nn_params, cost] = fmincg(costF, params , options);
%  [a b] = costFunction( (1:7*2305) ./(7*2305) ,[48*48 7],trainX,trainY,0);
%  there is a need for normalization
