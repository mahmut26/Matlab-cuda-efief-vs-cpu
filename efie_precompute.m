function pre = efie_precompute(mesh, opts)

% ---- quad rules ----
[qp,  qw ] = tri_quad_rule(opts.quad_order);
if ~isfield(opts,'near_order'), opts.near_order = 4; end
[qp2, qw2] = tri_quad_rule(opts.near_order);

% ---- triangle geometry ----
tri = precompute_tri_geom(mesh);
Nt  = size(mesh.F,1);

pre.tri = tri;
pre.qp  = qp;  pre.qw  = qw;
pre.qp2 = qp2; pre.qw2 = qw2;

% ---- triangle length scale for near-test ----
% area2 = 2A  => A=0.5*area2
% l_eq = sqrt(4A/sqrt(3)) = sqrt(2*area2/sqrt(3))
htri = sqrt(2*tri.area2 / sqrt(3));
pre.htri = htri;

% ---- near matrix: nearMat(t1,t2) ----
% Nt küçük/orta ise NxN matris hızlı olur.
cx = tri.cent(:,1); cy = tri.cent(:,2); cz = tri.cent(:,3);
dx = cx - cx.'; dy = cy - cy.'; dz = cz - cz.';
D  = sqrt(dx.^2 + dy.^2 + dz.^2);

Thresh = opts.near_sing_eps * 0.5 * (htri + htri.');
pre.nearMat = (D < Thresh);

% ---- regular quadrature cache per triangle ----
Nq = size(qp,1);
pre.r_reg = zeros(Nq,3,Nt);
pre.w_reg = zeros(Nq,Nt);  % already includes Jacobian (area2)
for t = 1:Nt
    pre.r_reg(:,:,t) = map_ref_to_tri(tri, t, qp);
    pre.w_reg(:,t)   = qw(:) * tri.area2(t);   % (qw*J)
end

% ---- aeq per triangle for near softening ----
Atri = 0.5 * tri.area2;            % actual area
pre.aeqTri = sqrt(Atri/pi);

% ---- subdiv quadrature caches (self + near) ----
if ~isfield(opts,'self_subdiv'), opts.self_subdiv = 2; end
if ~isfield(opts,'self_alpha'),  opts.self_alpha  = 0.20; end
if ~isfield(opts,'near_subdiv'), opts.near_subdiv = 2; end
if ~isfield(opts,'near_alpha'),  opts.near_alpha  = 0.15; end

Nq2 = size(qp2,1);

pre.self = cell(Nt,1);
pre.near = cell(Nt,1);

for t = 1:Nt
    A = tri.r1(t,:); B = tri.r2(t,:); C = tri.r3(t,:);

    % ---- self subdiv cache ----
    subS = subdivide_triangles(A,B,C, opts.self_subdiv);
    NsS  = size(subS,3);
    rS   = zeros(Nq2,3,NsS);
    wS   = zeros(Nq2,NsS);
    aSoftS = zeros(1,NsS);

    for p = 1:NsS
        Ap = subS(1,:,p); Bp=subS(2,:,p); Cp=subS(3,:,p);
        a2p = norm(cross(Bp-Ap, Cp-Ap));       % 2A_sub
        rS(:,:,p) = map_ref_to_vertices(Ap,Bp,Cp, qp2);
        wS(:,p)   = qw2(:) * a2p;

        Asub = 0.5 * a2p;
        aeq  = sqrt(Asub/pi);
        aSoftS(p) = opts.self_alpha * aeq;
    end

    pre.self{t}.r = rS;
    pre.self{t}.w = wS;
    pre.self{t}.aSoft = aSoftS;

    % ---- near subdiv cache ----
    subN = subdivide_triangles(A,B,C, opts.near_subdiv);
    NsN  = size(subN,3);
    rN   = zeros(Nq2,3,NsN);
    wN   = zeros(Nq2,NsN);

    for p = 1:NsN
        Ap = subN(1,:,p); Bp=subN(2,:,p); Cp=subN(3,:,p);
        a2p = norm(cross(Bp-Ap, Cp-Ap));
        rN(:,:,p) = map_ref_to_vertices(Ap,Bp,Cp, qp2);
        wN(:,p)   = qw2(:) * a2p;
    end

    pre.near{t}.r = rN;
    pre.near{t}.w = wN;
end

end
