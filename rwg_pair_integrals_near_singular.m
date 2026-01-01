function [Ivec, Isca] = rwg_pair_integrals_near_singular(mesh, tri, rwg, m, n, t1, t2, qp, qw, k, opts)

if ~isfield(opts,'near_subdiv'), opts.near_subdiv=2; end
if ~isfield(opts,'near_alpha'),  opts.near_alpha =0.15; end
if ~isfield(opts,'near_order'),  opts.near_order =4; end

[qp2, qw2] = tri_quad_rule(opts.near_order);

A1 = tri.r1(t1,:); B1 = tri.r2(t1,:); C1 = tri.r3(t1,:);
A2 = tri.r1(t2,:); B2 = tri.r2(t2,:); C2 = tri.r3(t2,:);

sub1 = subdivide_triangles(A1,B1,C1, opts.near_subdiv);
sub2 = subdivide_triangles(A2,B2,C2, opts.near_subdiv);
Ns1 = size(sub1,3); Ns2 = size(sub2,3);

Atri1 = 0.5*tri.area2(t1);
Atri2 = 0.5*tri.area2(t2);
aeq = min(sqrt(Atri1/pi), sqrt(Atri2/pi));
a_soft = opts.near_alpha * aeq;

Ivec = complex(0,0);
Isca = complex(0,0);

for p=1:Ns1
    Ap=sub1(1,:,p); Bp=sub1(2,:,p); Cp=sub1(3,:,p);
    a2p = norm(cross(Bp-Ap, Cp-Ap));
    r  = map_ref_to_vertices(Ap,Bp,Cp, qp2);
    wp = qw2(:)*a2p;
    [fm, divm] = rwg_eval_on_triangle(rwg, m, t1, r);

    for q=1:Ns2
        Aq=sub2(1,:,q); Bq=sub2(2,:,q); Cq=sub2(3,:,q);
        a2q = norm(cross(Bq-Aq, Cq-Aq));
        rp = map_ref_to_vertices(Aq,Bq,Cq, qp2);
        wq = qw2(:)*a2q;
        [fn, divn] = rwg_eval_on_triangle(rwg, n, t2, rp);

        dx = r(:,1) - rp(:,1).';
        dy = r(:,2) - rp(:,2).';
        dz = r(:,3) - rp(:,3).';
        R  = sqrt(dx.^2 + dy.^2 + dz.^2);
        R  = sqrt(R.^2 + a_soft^2);

        G = exp(-1j*k.*R)./(4*pi*R);

        FmFn   = fm(:,1)*fn(:,1).' + fm(:,2)*fn(:,2).' + fm(:,3)*fn(:,3).';
        DivDiv = divm(:)*divn(:).';
        W      = wp(:)*wq(:).';

        Ivec = Ivec + sum(sum(W .* FmFn  .* G));
        Isca = Isca + sum(sum(W .* DivDiv .* G));
    end
end

end

function r = map_ref_to_vertices(A,B,C, qp)
u = qp(:,1); v = qp(:,2);
r = A + u.*(B-A) + v.*(C-A);
end

function subTris = subdivide_triangles(A,B,C, depth)
subTris = cat(3, [A;B;C]);
for d=1:depth
    old = subTris;
    Ns0 = size(old,3);
    new = zeros(3,3,4*Ns0);
    t=1;
    for s=1:Ns0
        A0=old(1,:,s); B0=old(2,:,s); C0=old(3,:,s);
        AB=0.5*(A0+B0);
        BC=0.5*(B0+C0);
        CA=0.5*(C0+A0);
        new(:,:,t)=[A0;AB;CA]; t=t+1;
        new(:,:,t)=[AB;B0;BC]; t=t+1;
        new(:,:,t)=[CA;BC;C0]; t=t+1;
        new(:,:,t)=[AB;BC;CA]; t=t+1;
    end
    subTris = new;
end
end
