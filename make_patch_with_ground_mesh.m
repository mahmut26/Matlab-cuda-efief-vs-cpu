function [V,F,patchFaceIdx,groundFaceIdx] = make_patch_with_ground_mesh( ...
                Lp, Wp, Lg, Wg, h, Np_x, Np_y, Ng_x, Ng_y)

% Patch (üst): x in [-Wp/2, Wp/2], y in [-Lp/2, Lp/2], z=+h/2
[xp, yp] = meshgrid( linspace(-Wp/2, Wp/2, Np_x+1), ...
                     linspace(-Lp/2, Lp/2, Np_y+1) );
Vp = [xp(:), yp(:), (+h/2)*ones(numel(xp),1)];

Fp = zeros(2*Np_x*Np_y,3);
t = 1;
for j=1:Np_y
    for i=1:Np_x
        v1 = (j-1)*(Np_x+1)+i;
        v2 = v1+1;
        v3 = v1+(Np_x+1);
        v4 = v3+1;
        Fp(t,:) = [v1 v2 v4]; t=t+1;
        Fp(t,:) = [v1 v4 v3]; t=t+1;
    end
end

% Ground (alt): x in [-Wg/2, Wg/2], y in [-Lg/2, Lg/2], z=-h/2
[xg, yg] = meshgrid( linspace(-Wg/2, Wg/2, Ng_x+1), ...
                     linspace(-Lg/2, Lg/2, Ng_y+1) );
Vg = [xg(:), yg(:), (-h/2)*ones(numel(xg),1)];

Fg = zeros(2*Ng_x*Ng_y,3);
offset = size(Vp,1);
t = 1;
for j=1:Ng_y
    for i=1:Ng_x
        v1 = offset + (j-1)*(Ng_x+1)+i;
        v2 = v1+1;
        v3 = v1+(Ng_x+1);
        v4 = v3+1;
        Fg(t,:) = [v1 v2 v4]; t=t+1;
        Fg(t,:) = [v1 v4 v3]; t=t+1;
    end
end

V = [Vp; Vg];
F = [Fp; Fg];

patchFaceIdx  = 1:size(Fp,1);
groundFaceIdx = size(Fp,1) + (1:size(Fg,1));
end
