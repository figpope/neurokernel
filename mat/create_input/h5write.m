function h5write(data, filename)
% DO NOT store complex valued data using this

hdf5write(filename, '/array',data);
