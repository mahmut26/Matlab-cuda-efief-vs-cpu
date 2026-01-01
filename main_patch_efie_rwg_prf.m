function main_patch_efie_rwg_prf()
clc; clear; close all;

%% ===== Fizik =====
c   = 299792458;
mu0 = 4*pi*1e-7;
eps0 = 1/(mu0*c^2);

%% ===== Frekans süpürmesi =====
freqs = linspace(1.0e9, 1.4e9, 41);
N     = numel(freqs);

%% ===== Substrat/etkin epsilon =====
er   = 2.59;
tand = 1e-4;

%% ===== Geometri =====
Lp = 20.1e-3;   % y (length)
Wp = 20.1e-3;   % x (width)
Lg = 0.20;      % y
Wg = 0.20;      % x
h  = 1.59e-3;   % patch-ground separation

% microstrip epsr_eff (bu modelde f'den bağımsız varsayılıyor)
epsr_eff = (er+1)/2 + (er-1)/2 * (1 + 12*h/Wp)^(-0.5);
eps_eff  = eps0 * epsr_eff * (1 - 1j*tand);

fprintf("epsr_eff=%.6f (Re)\n", real(epsr_eff));

%% ===== Mesh çözünürlüğü =====
Np_x = 12;  Np_y = 8;    % patch
Ng_x = 18;  Ng_y = 18;   % ground

[V,F,~,~] = make_patch_with_ground_mesh( ...
    Lp, Wp, Lg, Wg, h, Np_x, Np_y, Ng_x, Ng_y);

mesh.V = V;
mesh.F = F;

fprintf("Vertices=%d | Triangles=%d\n", size(V,1), size(F,1));

%% ===== RWG =====
rwg = build_rwg_interior(mesh);   % interior-only, orientation-aware
fprintf("RWG unknowns=%d\n", rwg.Ne);

%% ===== Feed noktası =====
insetDepth = 13.2e-3;

feed_x = -Wp/2 + 1e-3;
feed_y = -Lp/2 + insetDepth;

portPointPatch = [feed_x, feed_y, +h/2];
portPointGnd   = [feed_x, feed_y, -h/2];

portEdgePatch = pick_port_edge_by_point_and_layer(rwg, portPointPatch, +1);
portEdgeGnd   = pick_port_edge_by_point_and_layer(rwg, portPointGnd,   -1);

fprintf("portEdgePatch=%d | portEdgeGnd=%d\n", portEdgePatch, portEdgeGnd);

%% ===== Uyarım (diferansiyel edge-strip) =====
exc.V0 = 1.0;
exc.patchEdge  = portEdgePatch;
exc.groundEdge = portEdgeGnd;
exc.spreadRadius = 1.5 * mean(rwg.len);

% Port ağırlıkları (geometriye bağlı; bir kere)
[wPatch, wGnd] = port_weights_gaussian(mesh, rwg, exc.patchEdge, exc.groundEdge, exc.spreadRadius);

%% ===== Numerik ayarlar =====
opts.quad_order     = 4;
opts.near_sing_eps  = 2.5;

opts.self_subdiv = 2;
opts.self_alpha  = 0.20;

opts.near_subdiv = 2;
opts.near_alpha  = 0.15;
opts.near_order  = 4;

%% ===== "Üstten-alttan" parfor sırası =====
% order = [1, N, 2, N-1, 3, ...]
order = zeros(1,N);
lo = 1; hi = N;
for t = 1:N
    if mod(t,2)==1
        order(t) = lo; lo = lo + 1;   % üstten
    else
        order(t) = hi; hi = hi - 1;   % alttan
    end
end

%% ===== parfor çıktıları (t ile yazılacak) =====
Zin_tmp   = complex(zeros(1,N));
Iport_tmp = complex(zeros(1,N));

% İstersen burada bir pool açabilirsin:
% if isempty(gcp('nocreate')), parpool; end

%% ===== Frekans döngüsü (parfor) =====
parfor t = 1:N
    idx = order(t);        % gerçek frekans indeksi
    f = freqs(idx);
    omega = 2*pi*f;

    % tutarlı k:
    k = omega * sqrt(mu0 * eps_eff);

    [Z, Vrhs] = assemble_ZV_EFIE_RWG_DIFF_CPU(mesh, rwg, k, omega, mu0, eps_eff, exc, opts);

    I = Z \ Vrhs;

    Ipatch = sum( (wPatch(:) .* I(:)) .* rwg.len(:) );
    Ignd   = sum( (wGnd(:)   .* I(:)) .* rwg.len(:) );
    Iport_here = Ipatch - Ignd;

    Iport_tmp(t) = Iport_here;
    Zin_tmp(t)   = exc.V0 / Iport_here;
end

%% ===== Sonuçları doğru frekans indekslerine yerleştir =====
Zin   = complex(zeros(1,N));
Iport = complex(zeros(1,N));

Zin(order)   = Zin_tmp;
Iport(order) = Iport_tmp;

%% ===== Rapor =====
fprintf("\n=== Sweep sonuçları (özet) ===\n");
fprintf("f_start=%.3f GHz | f_stop=%.3f GHz | N=%d\n", freqs(1)/1e9, freqs(end)/1e9, N);

% İstersen hepsini yazdır:
for i = 1:N
    fprintf("f=%.3f GHz  Zin = %.6f %+.6fj Ohm\n", freqs(i)/1e9, real(Zin(i)), imag(Zin(i)));
end

%% ===== Plot =====
figure;
plot(freqs/1e9, real(Zin), '-o'); hold on;
plot(freqs/1e9, imag(Zin), '-s'); grid on;
xlabel('f (GHz)'); ylabel('Z_{in} (\Omega)');
legend('Re\{Z_{in}\}', 'Im\{Z_{in}\}', 'Location','best');
title('Giriş Empedansı Frekans Süpürmesi');

% (Opsiyonel) S11 (50 ohm)
% Z0  = 50;
% S11 = (Zin - Z0) ./ (Zin + Z0);
% figure; plot(freqs/1e9, 20*log10(abs(S11)), '-o'); grid on;
% xlabel('f (GHz)'); ylabel('|S_{11}| (dB)'); title('S11');

end
