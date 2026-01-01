function [Ivec, Isca] = rwg_pair_integrals_regular_fast(pre, rwg, m, n, t1, t2, k)

r  = pre.r_reg(:,:,t1);
rp = pre.r_reg(:,:,t2);

w1 = pre.w_reg(:,t1);
w2 = pre.w_reg(:,t2);
W  = w1 * w2.';  % (Nq x Nq) weights

[fm, divm] = rwg_eval_on_triangle(rwg, m, t1, r);
[fn, divn] = rwg_eval_on_triangle(rwg, n, t2, rp);

dx = r(:,1) - rp(:,1).';
dy = r(:,2) - rp(:,2).';
dz = r(:,3) - rp(:,3).';
R  = sqrt(dx.^2 + dy.^2 + dz.^2);
R  = max(R, 1e-15);

G = exp(-1j*k.*R)./(4*pi*R);

FmFn   = fm * fn.';           % dot products matrix
DivDiv = divm(:) * divn(:).';

Ivec = sum(sum(W .* FmFn  .* G));
Isca = sum(sum(W .* DivDiv .* G));

end

function [Ivec, Isca] = rwg_pair_integrals_near_cached(pre, rwg, m, n, t1, t2, k, opts)

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

function [Ivec, Isca] = rwg_pair_integrals_singular_cached(pre, rwg, m, n, t, k)

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
