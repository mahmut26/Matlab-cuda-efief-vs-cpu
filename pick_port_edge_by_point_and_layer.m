function eidx = pick_port_edge_by_point_and_layer(rwg, p, layerSign)
mask = (rwg.zSign == layerSign);
if ~any(mask)
    error("Bu katmanda edge yok (layerSign=%d).", layerSign);
end
cent = rwg.center(mask,:);
d = vecnorm(cent - p, 2, 2);
idx = find(mask);
[~,ii] = min(d);
eidx = idx(ii);
end
