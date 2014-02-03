#
# Performs initial preprocessing. Parses csv representation and creates
# corresponding matrix. Converts input file to octave matrix file.

# type of image
train = 1;
public = 2;
private = 3;


# all examples
m = 35887;

# 48*48 pixels in the image
d = 48;

# nr of pixels + visibility + label
A = zeros(m,d*d+2);

i = 0;
fid = fopen('../data/fer2013.csv','r');
percent = '%';
if ( fid < 0 )
	printf('Error: could not open file\n')
else
        pixels = zeros(d*d,1);
	while ~feof(fid), line = fgetl(fid);
		if ( i == 0 )
			i = i + 1;
			continue;
		end;
		if ( i < 30000 ) 
			i = i+1;
			continue;
		end;
		
		[label values type] = strsplit(line,','){1,:};
		dlabel = str2num(label);
		svalues = strsplit(values,' ');
#		printf('%d\n',vsize);
		for k = 1:d*d
			p = svalues{1,k};
			pixels(k) = str2num(p);
#			printf('%s %d\n',p,pixels(k));
		end
#		printf('\n');
		
		t = 0;
		if ( strcmp('Training',type))
			t = train;
		elseif ( strcmp('PublicTest',type))
			t = public;
		elseif ( strcmp('PrivateTest',type))
			t = private;
		end;
		if ( mod(i,100) == 0 )
			printf('Image: %d %.2f%s\n',i,100.0*i/m,percent);	
			fflush(stdout);
		end;
		A(i,1) = dlabel;
		A(i,d*d+2) = t;
		A(i,2:(d*d+1)) = pixels(:);
		i = i + 1;
	end;

	fclose(fid);
	save('../data/All.mat','A');
end;


