function main_patch_mex_sweep()
clc; clear; close all;

%% ===== Fizik =====
c   = 299792458;
mu0 = 4*pi*1e-7;
eps0 = 1/(mu0*c^2);

%% ===== Substrat =====
er   = 2.59;
tand = 1e-4;

%% ===== Geometri =====
Lp = 20.1e-3;   % y (length)
Wp = 20.1e-3;   % x (width)
Lg = 0.20;      % y
Wg = 0.20;      % x
h  = 1.59e-3;   % separation

% microstrip epsr_eff
epsr_eff = (er+1)/2 + (er-1)/2 * (1 + 12*h/Wp)^(-0.5);
eps_eff  = eps0 * epsr_eff * (1 - 1j*tand);

fprintf("epsr_eff=%.4f, tand=%.2g\n", real(epsr_eff), tand);

%% ===== Mesh =====
Np_x = 12;  Np_y = 8;
Ng_x = 18;  Ng_y = 18;

[V,F] = make_patch_with_ground_mesh(Lp, Wp, Lg, Wg, h, Np_x, Np_y, Ng_x, Ng_y);
mesh.V = V;
mesh.F = int32(F);   % MEX int32 istiyor

fprintf("Vertices=%d | Triangles=%d\n", size(V,1), size(F,1));

%% ===== RWG (interior-only, orientation-aware) =====
rwg = build_rwg_interior(mesh);

% MEX için int32 cast
rwg.edge      = int32(rwg.edge);
rwg.plusTri   = int32(rwg.plusTri);
rwg.minusTri  = int32(rwg.minusTri);
rwg.plusSign  = int32(rwg.plusSign);
rwg.minusSign = int32(rwg.minusSign);

fprintf("RWG unknowns=%d\n", rwg.Ne);

%% ===== Feed (sen seçiyorsun) =====
insetDepth = 13.2e-3;
feed_x = -Wp/2 + 1e-3;
feed_y = -Lp/2 + insetDepth;

portPointPatch = [feed_x, feed_y, +h/2];
portPointGnd   = [feed_x, feed_y, -h/2];

portEdgePatch = pick_port_edge_by_point_and_layer(rwg, portPointPatch, +1);
portEdgeGnd   = pick_port_edge_by_point_and_layer(rwg, portPointGnd,   -1);

fprintf("portEdgePatch=%d | portEdgeGnd=%d\n", portEdgePatch, portEdgeGnd);

%% ===== Sweep =====
freqs = linspace(1.0e9, 1.4e9, 41).';   % Nf x 1
V0 = 1.0;
spreadRadius = 1.5 * mean(rwg.len);
Z0 = 50;

% === GPU MEX çağrısı ===
[Zin, S11] = patch_sweep_cuda( ...
    mesh.V, mesh.F, rwg.edge, ...
    rwg.plusTri, rwg.minusTri, ...
    rwg.plusSign, rwg.minusSign, ...
    rwg.rp, rwg.rm, rwg.Ap, rwg.Am, rwg.len, ...
    freqs, mu0, eps_eff, ...
    int32(portEdgePatch), int32(portEdgeGnd), ...
    V0, spreadRadius, Z0 );

%% ===== Plot =====
figure('Color','w');
subplot(2,1,1);
plot(freqs/1e9, real(Zin), 'LineWidth', 2); hold on;
plot(freqs/1e9, imag(Zin), '--', 'LineWidth', 2);
grid on; xlabel('GHz'); ylabel('Z_{in} (Ohm)');
legend('Re','Im'); title('Zin (GPU MEX, centroid approx)');

subplot(2,1,2);
plot(freqs/1e9, 20*log10(abs(S11)), 'LineWidth', 2);
grid on; xlabel('GHz'); ylabel('|S11| (dB)');
title('|S11| (GPU MEX, centroid approx)');
end
