function [display_array] = drawImage(X,d1,d2)

colormap(gray);

max_val = max(abs(X));
display_array = reshape(X,d1,d2)'/max_val;

imagesc(display_array, [-1 1]);

axis image off;
drawnow;

end;

