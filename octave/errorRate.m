function [error] = errorRate(Y, YOriginal)

error = 0;
error = mean(double(Y != YOriginal)) * 100;

end;
