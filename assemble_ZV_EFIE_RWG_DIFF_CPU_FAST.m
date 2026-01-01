function [Z, Vrhs] = assemble_ZV_EFIE_RWG_DIFF_CPU_FAST(mesh, rwg, k, omega, mu, eps, exc, opts, pre)
% FAST assemble: precompute cache (pre) kullanır.

Ne = rwg.Ne;
Z  = complex(zeros(Ne,Ne));

nearMat = pre.nearMat;

% opsiyonel: simetri ile yarım hesap
useSym = isfield(opts,'use_symmetry') && opts.use_symmetry;

if useSym
    for m = 1:Ne
        tm = [rwg.plusTri(m), rwg.minusTri(m)];
        for n = m:Ne
            tn = [rwg.plusTri(n), rwg.minusTri(n)];
            Zmn = Z_entry(rwg, pre, m, n, tm, tn, k, omega, mu, eps, opts, nearMat);
            Z(m,n) = Zmn;
            Z(n,m) = Zmn; % dikkat: her zaman güvenli olmayabilir
        end
    end
else
    for m = 1:Ne
        tm = [rwg.plusTri(m), rwg.minusTri(m)];
        for n = 1:Ne
            tn = [rwg.plusTri(n), rwg.minusTri(n)];
            Z(m,n) = Z_entry(rwg, pre, m, n, tm, tn, k, omega, mu, eps, opts, nearMat);
        end
    end
end

% RHS (senin mevcut fonksiyonların)
Vrhs = rhs_edge_strip_diff_gaussian(rwg, exc.patchEdge, exc.groundEdge, exc.V0, exc.spreadRadius);

end

% =====================================================================
function Zmn = Z_entry(rwg, pre, m, n, tm, tn, k, omega, mu, eps, opts, nearMat)
Zmn = complex(0,0);

for im = 1:2
    t1 = tm(im);
    if t1==0, continue; end
    for in = 1:2
        t2 = tn(in);
        if t2==0, continue; end

        if t1 == t2
            [Ivec, Isca] = rwg_pair_integrals_singular_cached(pre, rwg, m, n, t1, k);
        elseif nearMat(t1,t2)
            [Ivec, Isca] = rwg_pair_integrals_near_cached(pre, rwg, m, n, t1, t2, k, opts);
        else
            [Ivec, Isca] = rwg_pair_integrals_regular_fast(pre, rwg, m, n, t1, t2, k);
        end

        Zmn = Zmn + 1j*omega*mu*Ivec + (1/(1j*omega*eps))*Isca;
    end
end

end

% =====================================================================
function [Ivec, Isca] = rwg_pair_integrals_regular_fast(pre, rwg, m, n, t1, t2, k)
% regular (uzak) integral: vektörize

r  = pre.r_reg(:,:,t1);
rp = pre.r_reg(:,:,t2);

w1 = pre.w_reg(:,t1);
w2 = pre.w_reg(:,t2);
W  = w1 * w2.';  % (Nq x Nq)

[fm, divm] = rwg_eval_on_triangle(rwg, m, t1, r);
[fn, divn] = rwg_eval_on_triangle(rwg, n, t2, rp);

dx = r(:,1) - rp(:,1).';
dy = r(:,2) - rp(:,2).';
dz = r(:,3) - rp(:,3).';
R  = sqrt(dx.^2 + dy.^2 + dz.^2);
R  = max(R, 1e-15);

G = exp(-1j*k.*R)./(4*pi*R);

FmFn   = fm * fn.';                 % Nq x Nq
DivDiv = divm(:) * divn(:).';

Ivec = sum(sum(W .* FmFn  .* G));
Isca = sum(sum(W .* DivDiv .* G));
end

% =====================================================================
function [Ivec, Isca] = rwg_pair_integrals_near_cached(pre, rwg, m, n, t1, t2, k, opts)
% near integral: triangle başına subdiv+quad cache’lerini kullanır

data1 = pre.near{t1};
data2 = pre.near{t2};

rAll  = data1.r;   wAll  = data1.w;
rpAll = data2.r;   wqAll = data2.w;

Ns1 = size(rAll,3);
Ns2 = size(rpAll,3);

aeq = min(pre.aeqTri(t1), pre.aeqTri(t2));
a_soft = opts.near_alpha * aeq;

Ivec = complex(0,0);
Isca = complex(0,0);

for p = 1:Ns1
    r  = rAll(:,:,p);
    wp = wAll(:,p);
    [fm, divm] = rwg_eval_on_triangle(rwg, m, t1, r);

    for q = 1:Ns2
        rp = rpAll(:,:,q);
        wq = wqAll(:,q);
        [fn, divn] = rwg_eval_on_triangle(rwg, n, t2, rp);

        dx = r(:,1) - rp(:,1).';
        dy = r(:,2) - rp(:,2).';
        dz = r(:,3) - rp(:,3).';
        R  = sqrt(dx.^2 + dy.^2 + dz.^2);
        R  = sqrt(R.^2 + a_soft^2);

        G = exp(-1j*k.*R)./(4*pi*R);

        FmFn   = fm * fn.';
        DivDiv = divm(:) * divn(:).';
        W      = wp(:) * wq(:).';

        Ivec = Ivec + sum(sum(W .* FmFn  .* G));
        Isca = Isca + sum(sum(W .* DivDiv .* G));
    end
end
end

% =====================================================================
function [Ivec, Isca] = rwg_pair_integrals_singular_cached(pre, rwg, m, n, t, k)
% self integral: triangle başına subdiv+quad cache + p==q softening

data = pre.self{t};
rAll = data.r;
wAll = data.w;
aS   = data.aSoft;

Ns = size(rAll,3);

Ivec = complex(0,0);
Isca = complex(0,0);

for p = 1:Ns
    r  = rAll(:,:,p);
    wp = wAll(:,p);
    [fm, divm] = rwg_eval_on_triangle(rwg, m, t, r);

    for q = 1:Ns
        rp = rAll(:,:,q);
        wq = wAll(:,q);
        [fn, divn] = rwg_eval_on_triangle(rwg, n, t, rp);

        dx = r(:,1) - rp(:,1).';
        dy = r(:,2) - rp(:,2).';
        dz = r(:,3) - rp(:,3).';
        R  = sqrt(dx.^2 + dy.^2 + dz.^2);

        if p==q
            R = sqrt(R.^2 + aS(p)^2);
        else
            R = max(R, 1e-15);
        end

        G = exp(-1j*k.*R)./(4*pi*R);

        FmFn   = fm * fn.';
        DivDiv = divm(:) * divn(:).';
        W      = wp(:) * wq(:).';

        Ivec = Ivec + sum(sum(W .* FmFn  .* G));
        Isca = Isca + sum(sum(W .* DivDiv .* G));
    end
end
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
