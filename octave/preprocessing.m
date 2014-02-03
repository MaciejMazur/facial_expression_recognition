

# type of image
train = 0;
public = 1;
private = 2;


# all examples
m = 35887;

# 48*48 pixels in the image
d = 48;

# nr of pixels + visibility + label
A = zeros(m,d*d+2);

i = 0;
fid = fopen('../data/fer2013.csv','r');
if ( fid < 0 )
	printf('Error: could not open file\n')
else
	while ~feof(fid), line = fgetl(fid);
		if ( i == 0 )
			i = i + 1;
			continue;
		end;
		if ( i > 10 ) 
			break;
		end;
		printf('%d\n',i);
		i = i + 1;
	end;

	fclose(fid);
end;
