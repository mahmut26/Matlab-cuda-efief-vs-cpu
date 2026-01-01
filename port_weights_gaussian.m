function [wPatch, wGnd] = port_weights_gaussian(mesh, rwg, patchEdge, groundEdge, spreadRadius)

Ne = rwg.Ne;

% edge merkezleri
mid = zeros(Ne,3);
for e=1:Ne
    v1 = rwg.edge(e,1); v2 = rwg.edge(e,2);
    mid(e,:) = 0.5*(mesh.V(v1,:) + mesh.V(v2,:));
end

% layer belirleme (zSign varsa onu kullan)
zmid = 0.5*(min(mid(:,3)) + max(mid(:,3)));
layer = @(e) (mid(e,3) > zmid)*2 - 1; % +1/-1

% patch weights
r0 = mid(patchEdge,:);
mask = (arrayfun(layer,1:Ne).' == layer(patchEdge));
d = vecnorm(mid - r0, 2, 2);
wPatch = exp(-(d/spreadRadius).^2) .* mask;
wPatch = wPatch / sum(wPatch);

% ground weights
r0 = mid(groundEdge,:);
mask = (arrayfun(layer,1:Ne).' == layer(groundEdge));
d = vecnorm(mid - r0, 2, 2);
wGnd = exp(-(d/spreadRadius).^2) .* mask;
wGnd = wGnd / sum(wGnd);
end
