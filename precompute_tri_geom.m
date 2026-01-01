function tri = precompute_tri_geom(mesh)
V = mesh.V; F = mesh.F;
Nt = size(F,1);

tri.r1 = zeros(Nt,3);
tri.r2 = zeros(Nt,3);
tri.r3 = zeros(Nt,3);
tri.area2 = zeros(Nt,1);   % 2A
tri.cent = zeros(Nt,3);

for t=1:Nt
    ids = F(t,:);
    r1 = V(ids(1),:); r2 = V(ids(2),:); r3 = V(ids(3),:);
    tri.r1(t,:) = r1; tri.r2(t,:) = r2; tri.r3(t,:) = r3;
    tri.cent(t,:) = (r1+r2+r3)/3;
    tri.area2(t)  = norm(cross(r2-r1, r3-r1)); % 2A
end
end
