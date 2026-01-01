function [f, divf] = rwg_eval_on_triangle(rwg, m, t, r)
% r: [Nq x 3] points on triangle t
Nq = size(r,1);
f = zeros(Nq,3);
divf = zeros(Nq,1);

tp = rwg.plusTri(m);
tm = rwg.minusTri(m);

if t == tp
    s = rwg.plusSign(m);
    l = rwg.len(m);
    A = rwg.Ap(m);
    rp = rwg.rp(m,:);
    f = s * (l/(2*A)) * (r - rp);
    divf(:) = s * (l/A);
    return;
end

if t == tm
    s = rwg.minusSign(m);
    l = rwg.len(m);
    A = rwg.Am(m);
    rm = rwg.rm(m,:);
    f = s * (l/(2*A)) * (rm - r);
    divf(:) = s * (-l/A);
    return;
end
end
