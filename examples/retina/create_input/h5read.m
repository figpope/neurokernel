function A = h5read(filename)
% DO NOT read h5 files containing complex valued data.

if nargin < 1

[filename,pathname]=uigetfile({'*.h5'},'Select The Input File');
filename=[pathname filename];
end

info = hdf5info(filename);

A = hdf5read(info.GroupHierarchy.Datasets);
