function [ sg_params, cost ] = stochasticGradientAutoencoder(params, layers, trainX, lambda, delta, iterations)

S=['Iteration: '];
sg_params = params;
cost = zeros(size(trainX,1),1);
oldCost = 10.0;
oldGrad = zeros(size(params,1),1);
momentum = 0.9;
for j = 1:iterations
	oldGrad = oldGrad .* 0;
	for i = 1:size(trainX,1)
		[ cost(i), grad ] = autoencoderCost(sg_params,layers, trainX(i:i,:), lambda);
		oldGrad = momentum * oldGrad .- delta * grad;
		sg_params = sg_params .+ delta * grad; 
        	if ( mod(i,100) == 0 )
			fprintf('%s %4i | Cost: %4.6e\r',S, i, cost(i));
                	fflush(stdout);
        	end;
	end
	pred = predictAutoencoder(sg_params,layers,trainX);
	
	fprintf('%s %i Cost: %4.6e Error: %f\n',S,j, mean(cost), mean(mean(abs(pred.-trainX)>0.01))*100 );

	if ( mean(cost) > oldCost )
		delta = delta / 1.01;
		fprintf('New delta: %4.6e\n', delta);
	end;
	fflush(stdout);
	oldCost = mean(cost);
end;
