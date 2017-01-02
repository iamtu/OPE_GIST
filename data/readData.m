function [X,Y] = readData(fileName)
	disp(sprintf('Reading file %s', fileName));

	Y = [];

	i = [];
	j = [];
	s = [];
	fid = fopen(fileName);

	doc_count = 0;
	tline = fgetl(fid);
	while ischar(tline)
		doc_count = doc_count + 1;
		[y, doc] = strtok(tline);
		Y = [Y; str2num(y)];

		pair = sscanf(doc,'%d:%d');
		for inner_iter = 0 : length(pair)/2-1
			i = [i, doc_count];
			j = [j, pair(2*inner_iter+1)];
			s = [s, pair(2*inner_iter+2)];
		end
	    
	    tline = fgetl(fid);
	end
	fclose(fid);

	X = sparse(i,j,s);