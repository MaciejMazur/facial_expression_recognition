function [normX] = normalize(X)

normX = X;

mu = mean(normX);
sigma = std(normX);

for i = 1:size(X,1)
	normX(i,:) = (normX(i,:)-mu)/sigma;
end;


end;
