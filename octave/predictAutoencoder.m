function p = predictAutoencoder(params, layers_sizes,X)

thetas = size(layers_sizes,2)-1;

%	theta labels
theta_names = ['1';'2';'3';'4';'5';'6';'7';'8';'9';'10';'11';'12'];
theta_begin = 1;

%	Unrolling parameters, initialization of Theta matrixes, and Theta gradients
for i = 1:thetas
	prev = layers_sizes(i)+1;
	curr = layers_sizes(i+1);
	Theta.(theta_names(i,:)) = reshape(params(theta_begin:theta_begin-1+prev*curr), ...
					curr, prev);

	theta_begin = theta_begin + curr*prev;
endfor;

m = size(X, 1);




p = zeros(size(X, 1), 1);
hi = X;

for i = 1:thetas
	hi = sigmoid([ones(m, 1) hi] * Theta.(theta_names(i,:))');
end


p = hi;




end
