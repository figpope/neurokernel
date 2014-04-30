1. Use script1.m to generate the inputs to the photoreceptor. See comments inside for instruction.

2. After running script1.m, 3 files will be created.
   retina_input.h5 is the input you need to provide to your simulation
   You can ignore the other two files.

3. Two video files will be generated:
   rec_linear.mp4 shows the raw photon count image, it may appear darker.
   rec_gamma_correct.mp4 shows the corrected image suitable for you eye.
   The lower right screen in the video shows the block inside the red frame.
   The upper right screen shows the inputs to R1 neurons in the ommatidia array.

4. Use the LPU_retina class (a subclass of LPU) provided in LPU_retina.py will read in the retina_input.h5 file correctly in each step. The only thing you need to do is to access self.photon_input gpuarray, whose size equals to the number of photoreceptors. 

5. The values of number of photons are not in integers but double precision floating points. This is fine since you still need to use this to calculate the average number of photons absorbed by 30000 microvilli before getting integer number of photons as inputs to each microvilli. Don’t worry if the total number of photons absorbed by all the 30000 microvilli does not match the value in the input file. They should be close enough though.


