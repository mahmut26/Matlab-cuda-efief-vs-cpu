function [Z, Vrhs] = assemble_ZV_EFIE_RWG_DIFF_CPU(mesh, rwg, k, omega, mu, eps, exc, opts)

Ne = rwg.Ne;
Z = complex(zeros(Ne,Ne));

[qp, qw] = tri_quad_rule(opts.quad_order);
tri = precompute_tri_geom(mesh);

for m = 1:Ne
    tm = [rwg.plusTri(m), rwg.minusTri(m)];

    for n = 1:Ne
        tn = [rwg.plusTri(n), rwg.minusTri(n)];
        Zmn = complex(0,0);

        for im=1:2
            for in=1:2
                t1 = tm(im);
                t2 = tn(in);
                if t1==0 || t2==0, continue; end

                if t1 == t2
                    [Ivec, Isca] = rwg_pair_integrals_singular(mesh, tri, rwg, m, n, t1, t2, qp, qw, k, opts);
                elseif triangles_are_near(tri, t1, t2, opts.near_sing_eps)
                    [Ivec, Isca] = rwg_pair_integrals_near_singular(mesh, tri, rwg, m, n, t1, t2, qp, qw, k, opts);
                else
                    [Ivec, Isca] = rwg_pair_integrals_regular(mesh, tri, rwg, m, n, t1, t2, qp, qw, k);
                end

                Zmn = Zmn + 1j*omega*mu*Ivec + (1/(1j*omega*eps))*Isca;
            end
        end

        Z(m,n) = Zmn;
    end
end

% RHS: diferansiyel Gaussian edge-strip (basit/stabil)
Vrhs = rhs_edge_strip_diff_gaussian(rwg, exc.patchEdge, exc.groundEdge, exc.V0, exc.spreadRadius);

end

function V = rhs_edge_strip_diff_gaussian(rwg, patchEdge, groundEdge, V0, spreadRadius)
Ne = rwg.Ne;
V = complex(zeros(Ne,1));

% patch layer weight
V = V + local_gauss_on_layer(rwg, patchEdge, +V0/2, spreadRadius);
% ground layer weight
V = V + local_gauss_on_layer(rwg, groundEdge, -V0/2, spreadRadius);
end

function v = local_gauss_on_layer(rwg, edge0, amp, spreadRadius)
Ne = rwg.Ne;
v = complex(zeros(Ne,1));

layer = rwg.zSign(edge0);
r0 = rwg.center(edge0,:);

mask = (rwg.zSign == layer);
d = vecnorm(rwg.center - r0, 2, 2);

w = exp(-(d/spreadRadius).^2) .* mask;
sw = sum(w);
if sw < 1e-15
    error("spreadRadius çok küçük (katmanda ağırlık sıfır).");
end
w = w / sw;

v = amp * w;
end
