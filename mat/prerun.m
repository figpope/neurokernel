cd('create_input')
script1
retina_input = h5read('retina_inputs.h5');
cd('..')
save('working.mat', 'retina_input');

