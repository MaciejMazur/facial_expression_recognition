function [ sg_params, cost ] = stochasticGradient(params, layers, trainX, trainY, lambda, labels, delta, iterations,len)

S=['Iteration: '];
sg_params = params;
cost = zeros(size(trainX,1),1);
oldCost = 10.0;
oldGrad = zeros(size(params,1),1);
momentum = 0.9;
for j = 1:iterations
	oldGrad = oldGrad .* 0;
	for i = 1:(size(trainX,1)/len)
		[ cost(i), grad ] = costFunction(sg_params,layers, trainX(1+(i-1)*len:i*len,:), trainY(1+(i-1)*len:i*len,:), lambda, labels);
		oldGrad = momentum * oldGrad .+ delta * grad;
		sg_params = sg_params .- delta * grad; 
        	if ( mod(i,100) == 0 )
			fprintf('%s %4i | Cost: %4.6e\r',S, i, cost(i));
                	fflush(stdout);
        	end;
	end
	pred = predict(sg_params,layers,trainX);
	
	fprintf('%s %i Cost: %4.6e Error: %f\n',S,j, mean(cost), errorRate(pred,trainY) );

	if ( mean(cost) > oldCost )
		delta = delta / 2;
		fprintf('New delta: %4.6e\n', delta);
	end;
	fflush(stdout);
	oldCost = mean(cost);
end;
