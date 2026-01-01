function [Ivec, Isca] = rwg_pair_integrals_regular(mesh, tri, rwg, m, n, t1, t2, qp, qw, k)
if t1 == t2
    error("regular integral t1==t2 için kullanılmaz.");
end

r  = map_ref_to_tri(tri, t1, qp);
rp = map_ref_to_tri(tri, t2, qp);

J1 = tri.area2(t1);
J2 = tri.area2(t2);

[fm, divm] = rwg_eval_on_triangle(rwg, m, t1, r);
[fn, divn] = rwg_eval_on_triangle(rwg, n, t2, rp);

Ivec = complex(0,0);
Isca = complex(0,0);

for i=1:size(qp,1)
    for j=1:size(qp,1)
        R = norm(r(i,:)-rp(j,:));
        if R < 1e-15, R = 1e-15; end
        G = exp(-1j*k*R)/(4*pi*R);

        w = qw(i)*qw(j) * J1 * J2;
        Ivec = Ivec + w * (dot(fm(i,:), fn(j,:)) * G);
        Isca = Isca + w * (divm(i)*divn(j) * G);
    end
end
end
