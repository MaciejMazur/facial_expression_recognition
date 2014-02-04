

#0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral


load('../data/All.mat');
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
