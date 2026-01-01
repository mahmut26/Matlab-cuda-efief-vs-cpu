function main_patch_gpu_mex_sweep_l()
clear;

mu0  = 4*pi*1e-7;
c    = 299792458;
eps0 = 1/(mu0*c^2);

freqs = linspace(1.0e9, 1.4e9, 41).';

er   = 2.59;
tand = 1e-4;

Lp = 20.1e-3;   Wp = 20.1e-3;
Lg = 0.20;      Wg = 0.20;
h  = 1.59e-3;

epsr_eff = (er+1)/2 + (er-1)/2 * (1 + 12*h/Wp)^(-0.5);
eps_eff  = eps0 * epsr_eff * (1 - 1j*tand);

Np_x = 8;  Np_y = 6;
Ng_x = 12; Ng_y = 12;

[V,F] = make_patch_with_ground_mesh(Lp, Wp, Lg, Wg, h, Np_x, Np_y, Ng_x, Ng_y);
mesh.V = V;
mesh.F = int32(F);

rwg = build_rwg_interior(mesh);
rwg.edge      = int32(rwg.edge);
rwg.plusTri   = int32(rwg.plusTri);
rwg.minusTri  = int32(rwg.minusTri);
rwg.plusSign  = int32(rwg.plusSign);
rwg.minusSign = int32(rwg.minusSign);

% feed
insetDepth = 13.2e-3;
feed_x = -Wp/2 + 1e-3;
feed_y = -Lp/2 + insetDepth;

portPointPatch = [feed_x, feed_y, +h/2];
portPointGnd   = [feed_x, feed_y, -h/2];

portEdgePatch = pick_port_edge_by_point_and_layer(rwg, portPointPatch, +1);
portEdgeGnd   = pick_port_edge_by_point_and_layer(rwg, portPointGnd,   -1);

V0 = 1.0;
spreadRadius = 1.5 * mean(rwg.len);
Z0 = 50;

% CPU FAST precompute + pack
opts.quad_order     = 2;
opts.near_sing_eps  = 1.2;
opts.self_subdiv    = 2;
opts.self_alpha     = 0.20;
opts.near_subdiv    = 1;
opts.near_alpha     = 0.15;
opts.near_order     = 2;

pre = efie_precompute(mesh, opts);
prepack = efie_precompute_pack(pre, opts);

% ---- call ONE MEX (sweep+solve)
[Zin, S11] = patch_sweep_fastquad_cuda( ...
    mesh.V, mesh.F, rwg.edge, ...
    rwg.plusTri, rwg.minusTri, rwg.plusSign, rwg.minusSign, ...
    rwg.rp, rwg.rm, rwg.Ap, rwg.Am, double(rwg.len), ...
    prepack, ...
    freqs, ...
    mu0, eps_eff, ...
    double(portEdgePatch), double(portEdgeGnd), ...
    V0, spreadRadius, Z0 ); %#ok<ASGLU>

% plot
figure('Color','w');
subplot(2,1,1);
plot(freqs/1e9, real(Zin), 'LineWidth', 2); hold on;
plot(freqs/1e9, imag(Zin), '--', 'LineWidth', 2);
grid on; xlabel('GHz'); ylabel('Z_{in} (\Omega)'); legend('Re','Im');

subplot(2,1,2);
plot(freqs/1e9, 20*log10(abs(S11)), 'LineWidth', 2);
grid on; xlabel('GHz'); ylabel('|S11| (dB)');
end
