function rwg = build_rwg_interior(mesh)
V = mesh.V; F = mesh.F;
Nt = size(F,1);

% directed edges per triangle (triangle vertex order)
rawKey = zeros(Nt*3,2);     % undirected sorted
rawDir = zeros(Nt*3,2);     % directed as in triangle
rawTri = zeros(Nt*3,1);

p=1;
for t=1:Nt
    tri = F(t,:);
    ed = [tri(1) tri(2);
          tri(2) tri(3);
          tri(3) tri(1)];
    for le=1:3
        a = ed(le,1); b = ed(le,2);
        rawDir(p,:) = [a b];
        rawKey(p,:) = sort([a b]);
        rawTri(p)   = t;
        p=p+1;
    end
end

[ue,~,ic] = unique(rawKey,'rows');      % undirected unique edges
Ne_all = size(ue,1);
counts = accumarray(ic,1,[Ne_all 1]);

% interior-only: exactly two triangles share
isInt = (counts==2);
ueInt = ue(isInt,:);
mapOldToNew = zeros(Ne_all,1);
mapOldToNew(isInt) = 1:sum(isInt);

Ne = size(ueInt,1);

plusTri  = zeros(Ne,1);
minusTri = zeros(Ne,1);
plusSign = zeros(Ne,1);
minusSign= zeros(Ne,1);

% assign + and - incidences with orientation sign relative to global (a<b) order
for eOld=1:Ne_all
    if ~isInt(eOld), continue; end
    eNew = mapOldToNew(eOld);

    rows = find(ic==eOld);   % 2 rows
    a = ue(eOld,1); b = ue(eOld,2);

    % incidence 1 -> plus
    r1 = rows(1);
    t1 = rawTri(r1);
    d1 = rawDir(r1,:);
    s1 = +1; if ~(d1(1)==a && d1(2)==b), s1=-1; end

    % incidence 2 -> minus
    r2 = rows(2);
    t2 = rawTri(r2);
    d2 = rawDir(r2,:);
    s2 = +1; if ~(d2(1)==a && d2(2)==b), s2=-1; end

    plusTri(eNew)=t1; plusSign(eNew)=s1;
    minusTri(eNew)=t2; minusSign(eNew)=s2;
end

% geometry: len, center, rp/rm, Ap/Am
len = zeros(Ne,1);
center = zeros(Ne,3);
rp = zeros(Ne,3); rm = zeros(Ne,3);
Ap = zeros(Ne,1); Am = zeros(Ne,1);

for e=1:Ne
    va = ueInt(e,1); vb = ueInt(e,2);
    ra = V(va,:); rb = V(vb,:);
    len(e) = norm(rb-ra);
    center(e,:) = 0.5*(ra+rb);

    tp = plusTri(e); tm = minusTri(e);

    [oppP, areaP] = opposite_vertex_and_area(V, F(tp,:), va, vb);
    [oppM, areaM] = opposite_vertex_and_area(V, F(tm,:), va, vb);

    rp(e,:) = V(oppP,:);  Ap(e)=areaP;
    rm(e,:) = V(oppM,:);  Am(e)=areaM;
end

% layer sign by z of center
zmid = 0.5*(max(V(:,3))+min(V(:,3)));
zSign = -ones(Ne,1);
zSign(center(:,3) > zmid) = +1;

rwg.Ne = Ne;
rwg.edge = ueInt;
rwg.plusTri = plusTri; rwg.minusTri = minusTri;
rwg.plusSign = plusSign; rwg.minusSign = minusSign;

rwg.rp = rp; rwg.rm = rm;
rwg.Ap = Ap; rwg.Am = Am;
rwg.len = len;
rwg.center = center;
rwg.zSign = zSign;

end

function [opp, A] = opposite_vertex_and_area(V, tri, va, vb)
ids = tri(:).';
opp = ids(ids~=va & ids~=vb);
if numel(opp)~=1
    error("Opposite vertex bulunamadı.");
end
r1 = V(tri(1),:); r2 = V(tri(2),:); r3 = V(tri(3),:);
A = 0.5*norm(cross(r2-r1, r3-r1));
end
