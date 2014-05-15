%% Use this file to generate videos that you can use for your project

%% set the number of rows and columns of the ommatidia hexagon array

% The array is defined in the following fashion
%                         cols
%             1    2    3    4    5    6
%         1   x         x         x
%                  x         x         x
%         2   x         x         x
%                  x         x         x
% rows    3   x         x         x
%                  x         x         x
%         4   x         x         x
%                  x         x         x
% The above diagram defines ncols = 6 and nrows = 4
nrows = 12*2;
ncols = 16*2;

%% other setting

T = 1; % total duration of simulation
dt = 1e-4; % time step
Nsteps = round(T/dt);

image_number = 1; % choose which image you want to use.

% set still_image to 1 for still image and 0 for video.
still_image = 0;


% ---------------------------------------------------------
% All settings done
% You don't have to do anything after this line

%% create hexagon array, note that the position is rounded to fit into pixels
scale = 4;
[h_array, p_array] = create_hexagon_array(nrows, ncols, scale);

xpos = round(h_array.xpos);
ypos = round(h_array.ypos);
num_ommatidia = h_array.num;

xstart = min(min(xpos));
ystart = min(min(ypos));
xend = max(max(xpos));
yend = max(max(ypos));

plot(xpos, ypos, 'x', 'MarkerSize', 10)
set(gca,'YDir','reverse')
axis(gca, 'equal')

margin = 20; % crop 20 pixels around the array boundary

%% define center of receptive field for each photoreceptor

total_R = num_ommatidia * 6; % total number of photoreceptors

R4dep = scale*0.4;
PR_array = create_R_position_plain(h_array, R4dep);

R_xpos = round(PR_array.xpos);
R_ypos = round(PR_array.ypos);

R_xpos_max = max(R_xpos);
R_ypos_max = max(R_ypos);

cc=hsv(6);
hold on
for i = 1:6
plot(R_xpos(:,i),R_ypos(:,i),'o', 'color', cc(i,:))
end
set(gca,'YDir','reverse')
axis(gca, 'equal')
legend('R7','R1','R2','R3','R4','R5','R6')

%% load image and blurring

% load image. The photon were store for a duration of 1ms.
load(['image', int2str(image_number), '.mat']);
im = im * (dt/1e-3); 
[h,w] = size(im);


maxpix = max(max(im));
im_gamma = (im./maxpix).^(1/2.2);

[rfx,rfy] = meshgrid([-32:32],[-32:32]);
sigma = 4;
rf = exp(- (rfx.^2+rfy.^2)/(2*sigma^2)) / (2*pi*sigma^2);
im_blurred = conv2(im, rf, 'same');
im_blurred_gamma = (im_blurred./maxpix).^(1/2.2);


%%
randn('state', 100)
rand('state', 100)

output = zeros(total_R, Nsteps); % the final inputs to all photoreceptors in each time step.
pos_x_rec = zeros(Nsteps, 1); % position history
pos_y_rec = pos_x_rec; 


% set initial position on the image, in pixel.
% in the case of still image, this sets which part of the image will be the
% input to the retina.
pos_x = round(rand()*(w-2*margin-xend-2) + margin);
pos_y = round(rand()*(h-2*margin-yend-2) + margin);

if still_image
    vx = 0;
    vy = 0;
else
    vx = randn(1)*1000;
    vy = randn(1)*1000;
end

for i = 1:Nsteps
    pos_x = pos_x + vx*dt;
    pos_y = pos_y + vy*dt;
    
    if round(pos_x) + xend + margin >= w
        pos_x = w-margin-(xend-xstart);
        vx = -vx;
    end
    
    if round(pos_x) - margin <= 0
        pos_x = margin;
        vx = -vx;
    end
    
    if round(pos_y) + yend + margin >= h
        pos_y = h-margin-(yend-ystart);
        vy = -vy;
    end
    
    if round(pos_y) - margin <= 0
        pos_y = margin;
        vy = -vy;
    end
    
    pos_x_rec(i) = pos_x;
    pos_y_rec(i) = pos_y;
    
    roundx = round(pos_x);
    roundy = round(pos_y);
    modx = pos_x - roundx;
    mody = pos_y - roundy;
    
    crop = im_blurred(roundy+1-margin:roundy+yend+margin, roundx+1-margin:roundx+xend+margin);
    output(:,i) = reshape(crop(sub2ind(size(crop), R_ypos+margin, R_xpos+margin)), total_R, 1);

    if ~still_image
        % this implement a change of speed randomly over time
        if rand()< dt % on average changes speed every second
           vx = randn(1)*1000;
           fprintf('change of vx to %.2f in step %d\n', vx, i)
        end

        if rand()< dt % on average changes speed every second
           vy = randn(1)*1000;
           fprintf('change of vy to %.2f in step %d\n', vy, i)
        end
    end
    
    if mod(i,100) == 1
        fprintf('%d\n',i)
    end
end

%% store results

% use retina_inputs.h5 as the input file to your simulation.
% make sure you have the same number of rows and columns for your retina
% setting.
h5write(output, 'retina_inputs.h5');
% when read in python, retina_inputs.h5 gives an array of dimension:
% number of rows = Nsteps
% number of columns = total number of photoreceptors
%
% In each row (time step),
% the retina inputs are stored in the following sequence:
%
% row 1, column 1, R1
% row 1, column 2, R1
%        :
% row 1, column m, R1
% row n, column 1, R1
% row n, column 2, R1
%        :
% row n, column m, R1
% row 1, column 1, R2
% row 1, column 2, R2
%        :
% row 1, column m, R2
% row n, column 1, R2
% row n, column 2, R2
%        :
% row n, column m, R2
%        :
%        :
%        :
% The values are number of photon per millisecond. 

% These two files records the positions retina move across the image
h5write(pos_x_rec, 'pos_x_rec.h5');
h5write(pos_y_rec, 'pos_y_rec.h5');


%% generate video showing how retina move across image, in a gamma corrected fashion

fig1 = figure;
video_W = 1080;
video_L = 560;
set(fig1, 'position', [100, 100, video_W, video_L])
winsize = get(fig1,'Position');
winsize(1:2) = [0 0];
set(fig1,'NextPlot','replacechildren');

write_to_file = 1;
if write_to_file
    writerObj = VideoWriter('rec_gamma_corrected.mp4', 'MPEG-4');
    writerObj.FrameRate = 10;
    wtiterObj.Quality = 100;
    open(writerObj);
end


rectangle_frame_w = xend - xstart;
rectangle_frame_h = yend - ystart;
rectangle_frame_x = [reshape(ones((yend-ystart), 4) .* repmat([xstart:-1:xstart-3], [yend-ystart,1]), 4*(yend-ystart), 1);...
    repmat([xstart+1:xend]',[4,1]); ...
    reshape(ones((yend-ystart), 4) .* repmat([xend:xend+3], [yend-ystart,1]), 4*(yend-ystart), 1);...
    repmat([xend:-1:xstart+1]', [4,1])];
rectangle_frame_y = [repmat([ystart+1:yend]', [4,1]); ...
    reshape(ones((xend-xstart), 4) .* repmat([yend:yend+3], [xend-xstart,1]), 4*(xend-xstart), 1);...
    repmat([yend:-1:ystart+1]', [4,1]);...
    reshape(ones((xend-xstart), 4) .* repmat([ystart:-1:ystart-3], [xend-xstart,1]), 4*(xend-xstart), 1);];


skipsteps = round(0.01/dt);
for i = 1:skipsteps:Nsteps
    subplot('position', [0.01, 0.01, 768/video_W, 512/video_L])
    im_rec = repmat(im_gamma, [1,1,3]);
    
    
    im_rec(sub2ind([h,w], rectangle_frame_y+round(pos_y_rec(i)), rectangle_frame_x+round(pos_x_rec(i)) )) = 1;
    imshow(im_rec);
    
    
    subplot('position', [768/video_W + 0.05, 0.05, 200/video_W, ncols*1.5/(nrows*sqrt(3))*200/video_L])
    v = im_blurred_gamma(round(pos_y_rec(i))+1:round(pos_y_rec(i))+yend, round(pos_x_rec(i))+1:round(pos_x_rec(i))+xend);
    imshow(v);
    
    subplot('position', [768/video_W + 0.05, 0.51, 200/video_W, ncols*1.5/(nrows*sqrt(3))*200/video_L ])
    aa = reshape(output(1:ncols*nrows, i),[nrows, ncols]);
    imshow((aa./maxpix).^(1/2.2))
    
    drawnow
    
    if write_to_file
        frame = getframe(fig1,winsize);
        writeVideo(writerObj,frame);
    end
    pause(0.01)
end
if write_to_file
    close(writerObj);
end

%% generate an animation showing the images in linear photon count fashion
% this is the actual input to your simulation

fig2 = figure;
video_W = 1080;
video_L = 560;
set(fig2, 'position', [100, 100, video_W, video_L])
winsize = get(fig1,'Position');
winsize(1:2) = [0 0];
set(fig2,'NextPlot','replacechildren');

% set write_to_file to 1 to output the figures to a video file
% set write_to_file to 0 to show video only on screen
write_to_file = 1;
if write_to_file
    writerObj = VideoWriter('rec_linear.mp4', 'MPEG-4');
    writerObj.FrameRate = 10;
    wtiterObj.Quality = 100;
    open(writerObj);
end


rectangle_frame_w = xend - xstart;
rectangle_frame_h = yend - ystart;
rectangle_frame_x = [reshape(ones((yend-ystart), 4) .* repmat([xstart:-1:xstart-3], [yend-ystart,1]), 4*(yend-ystart), 1);...
    repmat([xstart+1:xend]',[4,1]); ...
    reshape(ones((yend-ystart), 4) .* repmat([xend:xend+3], [yend-ystart,1]), 4*(yend-ystart), 1);...
    repmat([xend:-1:xstart+1]', [4,1])];
rectangle_frame_y = [repmat([ystart+1:yend]', [4,1]); ...
    reshape(ones((xend-xstart), 4) .* repmat([yend:yend+3], [xend-xstart,1]), 4*(xend-xstart), 1);...
    repmat([yend:-1:ystart+1]', [4,1]);...
    reshape(ones((xend-xstart), 4) .* repmat([ystart:-1:ystart-3], [xend-xstart,1]), 4*(xend-xstart), 1);];


skipsteps = round(0.01/dt);

for i = 1:skipsteps:Nsteps
    subplot('position', [0.01, 0.01, 768/video_W, 512/video_L])
    im_rec = repmat(im, [1,1,3]);
    
    
    im_rec(sub2ind([h,w], rectangle_frame_y+round(pos_y_rec(i)), rectangle_frame_x+round(pos_x_rec(i)) )) = maxpix;
    imshow(im_rec./maxpix);
    
    
    subplot('position', [768/video_W + 0.05, 0.05, 200/video_W, ncols*1.5/(nrows*sqrt(3))*200/video_L])
    v = im_blurred(round(pos_y_rec(i))+1:round(pos_y_rec(i))+yend, round(pos_x_rec(i))+1:round(pos_x_rec(i))+xend);
    imshow(v./maxpix);
    
    subplot('position', [768/video_W + 0.05, 0.51, 200/video_W, ncols*1.5/(nrows*sqrt(3))*200/video_L ])
    aa = reshape(output(1:ncols*nrows, i),[nrows, ncols]);
    imshow(aa./maxpix)
    
    drawnow
    
    if write_to_file
        frame = getframe(fig1,winsize);
        writeVideo(writerObj,frame);
    end
    pause(0.01)
end
if write_to_file
    close(writerObj);
end

