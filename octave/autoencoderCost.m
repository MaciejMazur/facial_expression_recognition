function [Cost grad] = autoencoderCost(params, layers_sizes, X, lambda)
%	Evaluates cost and count gradient using backpropagation algorithm.
%	params - theta parameters
%	layers_size array of layers sizes [input_layer, hidden_layer_1, ..., hidden_layer_n, output_layer]
%	X - data matrix
%	y - original results
%	lambda - regularization parameter


%	number of thetas
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
	Theta_grad.(theta_names(i,:)) = zeros(size(Theta.(theta_names(i,:))));

	theta_begin = theta_begin + curr*prev;
endfor;


m = size(X, 1);
         
% 	cost function
Cost = 0;

% 	square of parameters
tc = 0;

for i = 1:thetas
	tc = tc + sum(sum((Theta.(theta_names(i,:)))(:,2:end).^2));
endfor;

%	forward propagation
ai = X;
Zi.(theta_names(1,:)) = [ones(m, 1) ai];

for i = 1:thetas
  ai = [ones(m, 1) ai];
  Ai.(theta_names(i,:)) = ai;
  zi = ai * (Theta.(theta_names(i,:)))';
  Zi.(theta_names(i+1,:)) = zi;
  ai = sigmoid(zi);
endfor;

%	error evaluation

[A,p] = max(ai,[],2);


b = abs(X .- ai) ;

Cost = (sum(sum(b)) + (tc)*lambda/2)/m;

%	back propagation

di = (ai .- X)';
for i = thetas:-1:2
  Theta_grad.(theta_names(i,:)) = di * Ai.(theta_names(i,:));
  di = Theta.(theta_names(i,:))(:,2:end)' * di .* sigmoidGradient(Zi.(theta_names(i,:)))';
endfor;

Theta_grad.(theta_names(1,:)) = di * Ai.(theta_names(1,:));


for i = 1:thetas
	Theta_grad.(theta_names(i,:)) = Theta_grad.(theta_names(i,:)) ./ m;
endfor;


for i = 1:thetas
	Theta_grad.(theta_names(i,:))(:,2:end) = Theta_grad.(theta_names(i,:))(:,2:end) .+ ...
			lambda/m*Theta.(theta_names(i,:))(:,2:end);;
endfor;

grad = [];
for i = 1:thetas
	grad = [grad ; Theta_grad.(theta_names(i,:))(:) ];
endfor;


end
