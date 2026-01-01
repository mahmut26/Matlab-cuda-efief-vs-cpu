function [Ivec, Isca] = rwg_pair_integrals_singular(mesh, tri, rwg, m, n, t1, t2, qp, qw, k, opts)
% t1==t2 self-term: subdivision + scaled softening
if t1 ~= t2
    error("singular routine t1==t2 içindir.");
end

if ~isfield(opts,'self_subdiv'), opts.self_subdiv=2; end
if ~isfield(opts,'self_alpha'),  opts.self_alpha =0.20; end
if ~isfield(opts,'near_order'),  opts.near_order =4; end

[qp2, qw2] = tri_quad_rule(opts.near_order);

A = tri.r1(t1,:); B = tri.r2(t1,:); C = tri.r3(t1,:);
sub = subdivide_triangles(A,B,C, opts.self_subdiv);
Ns = size(sub,3);

Ivec = complex(0,0);
Isca = complex(0,0);

for p=1:Ns
    Ap = sub(1,:,p); Bp=sub(2,:,p); Cp=sub(3,:,p);
    a2p = norm(cross(Bp-Ap, Cp-Ap)); % 2A
    rpnt = map_ref_to_vertices(Ap,Bp,Cp, qp2);
    wp = qw2(:)*a2p;

    [fm, divm] = rwg_eval_on_triangle(rwg, m, t1, rpnt);

    % self-softening length for this subtriangle
    Asub = 0.5*a2p;
    aeq = sqrt(Asub/pi);
    a_soft = opts.self_alpha * aeq;

    for q=1:Ns
        Aq = sub(1,:,q); Bq=sub(2,:,q); Cq=sub(3,:,q);
        a2q = norm(cross(Bq-Aq, Cq-Aq));
        rqnt = map_ref_to_vertices(Aq,Bq,Cq, qp2);
        wq = qw2(:)*a2q;

        [fn, divn] = rwg_eval_on_triangle(rwg, n, t1, rqnt);

        dx = rpnt(:,1) - rqnt(:,1).';
        dy = rpnt(:,2) - rqnt(:,2).';
        dz = rpnt(:,3) - rqnt(:,3).';
        R  = sqrt(dx.^2 + dy.^2 + dz.^2);

        if p==q
            R = sqrt(R.^2 + a_soft^2);
        else
            R = max(R, 1e-15);
        end

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
