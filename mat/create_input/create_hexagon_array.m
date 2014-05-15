function [array, padded_array] = create_hexagon_array(nrow, ncol, ratio)

X = repmat([0:ncol-1]*1.5,[nrow,1]);
Y = repmat([0:nrow-1]'*sqrt(3),[1,ncol]);
col = repmat([0:ncol-1],[nrow,1]); %start from col 0
row = repmat([0:nrow-1]', [1,ncol]); %start from row 0

Y = Y + repmat([0 sqrt(3)/2],[nrow,ncol/2]);

X = reshape(X,nrow*ncol,1);
Y = reshape(Y,nrow*ncol,1);

col = reshape(col, nrow * ncol,1);
row = reshape(row, nrow * ncol,1); 

array = struct('xpos', X*ratio, 'ypos', Y*ratio, 'ncol', ncol, 'nrow', nrow, ...
    'col', col, 'row', row, 'num', ncol * nrow);

X = repmat([-2:ncol-1+2]*1.5,[nrow+4,1]);
Y = repmat([-2:nrow-1+2]'*sqrt(3),[1,ncol+4]);
col = repmat([-2:ncol-1+2],[nrow+4,1]); %start from col 0
row = repmat([-2:nrow-1+2]', [1,ncol+4]); %start from row 0

Y = Y + repmat([0 sqrt(3)/2],[nrow+4,(ncol+4)/2]);

X = reshape(X,(nrow+4)*(ncol+4),1);
Y = reshape(Y,(nrow+4)*(ncol+4),1);

col = reshape(col, (nrow+4) * (ncol+4),1);
row = reshape(row, (nrow+4) * (ncol+4),1); 

padded_array = struct('xpos', X*ratio, 'ypos', Y*ratio, 'ncol', ncol+4, 'nrow', nrow+4, ...
    'col', col, 'row', row, 'num', (ncol+4) * (nrow+4));


