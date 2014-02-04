function [error] = errorRate(Y, YOriginal)

error = 0;
error = sum(Y != YOriginal)/size(YOriginal,1);

end;
