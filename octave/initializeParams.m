function [params] = initializeParams(layers)

s = 0;

for i = 1:size(layers,2)-1
	s = s + (layers(i) + 1) * layers(i+1);
end

params = [0.5.* sin(pi*((1:s)/300).^2).*sin(2*pi *(1:s)/300)]';

%params = rand(s,1).- 0.5;


end
