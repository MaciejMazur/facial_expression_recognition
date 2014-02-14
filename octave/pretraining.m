function [ps] = pretraining(params,layers_sizes,X,y,lambda,labels, delta, iterations)


thetas = size(layers_sizes,2)-1;

%       theta labels
theta_begin = 1;

ps = [];
%       Unrolling parameters, initialization of Theta matrixes, and Theta gradients
for i = 1:thetas
        prev = layers_sizes(i)+1;
        curr = layers_sizes(i+1);
        t = reshape(params(theta_begin:theta_begin-1+prev*curr), ...
                                        curr, prev);
	
	p = initializeParams([prev-1 curr prev-1);
	[p cost] = stochasticGradientAutoencoder(p,X,lambda,delta,iterations);  
	ps = [ps; t];

        theta_begin = theta_begin + curr*prev;

endfor;
  
end
