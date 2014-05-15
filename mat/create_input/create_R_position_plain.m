function PR_array = create_R_position_plain(h_array, R4dep)

n_ommatidia = h_array.num;
xpos = zeros(n_ommatidia, 6);
ypos = zeros(n_ommatidia, 6);


xdeposition = [-sqrt(3)/2, -sqrt(3)/2, -sqrt(3)/2, 0, sqrt(3)/2, sqrt(3)/2]*R4dep;
ydeposition = -[-0.5, 0.5, 1.5, 1, 0.5, -0.5]*R4dep;



xpos = repmat(h_array.xpos, [1,6]) + repmat(xdeposition, [n_ommatidia, 1]);
ypos = repmat(h_array.ypos, [1,6]) + repmat(ydeposition, [n_ommatidia, 1]);

PR_array = struct('xpos', xpos, 'ypos', ypos);

