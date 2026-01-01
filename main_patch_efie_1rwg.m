function main_patch_efie_1rwg()
clc; clear; close all;

%% ===== Fizik =====
c   = 299792458;
mu0 = 4*pi*1e-7;
eps0 = 1/(mu0*c^2);

%% ===== Frekans süpürmesi =====
freqs = linspace(1.0e9, 1.4e9, 41);
N = numel(freqs);

%% ===== Substrat/etkin epsilon =====
er   = 2.59;
tand = 1e-4;

%% ===== Geometri =====
Lp = 20.1e-3; Wp = 20.1e-3;
Lg = 0.20;    Wg = 0.20;
h  = 1.59e-3;

epsr_eff = (er+1)/2 + (er-1)/2 * (1 + 12*h/Wp)^(-0.5);
eps_eff  = eps0 * epsr_eff * (1 - 1j*tand);

%% ===== Mesh =====
Np_x = 12;  Np_y = 8;
Ng_x = 18;  Ng_y = 18;

[V,F] = make_patch_with_ground_mesh(Lp, Wp, Lg, Wg, h, Np_x, Np_y, Ng_x, Ng_y);
mesh.V = V;
mesh.F = F;

fprintf("Vertices=%d | Triangles=%d\n", size(V,1), size(F,1));

%% ===== RWG =====
rwg = build_rwg_interior(mesh);
fprintf("RWG unknowns=%d\n", rwg.Ne);

%% ===== Feed =====
insetDepth = 13.2e-3;
feed_x = -Wp/2 + 1e-3;
feed_y = -Lp/2 + insetDepth;

portPointPatch = [feed_x, feed_y, +h/2];
portPointGnd   = [feed_x, feed_y, -h/2];

portEdgePatch = pick_port_edge_by_point_and_layer(rwg, portPointPatch, +1);
portEdgeGnd   = pick_port_edge_by_point_and_layer(rwg, portPointGnd,   -1);

fprintf("portEdgePatch=%d | portEdgeGnd=%d\n", portEdgePatch, portEdgeGnd);

%% ===== Uyarım =====
exc.V0 = 1.0;
exc.patchEdge  = portEdgePatch;
exc.groundEdge = portEdgeGnd;
exc.spreadRadius = 1.5 * mean(rwg.len);

[wPatch, wGnd] = port_weights_gaussian(mesh, rwg, exc.patchEdge, exc.groundEdge, exc.spreadRadius);

%% ===== Numerik ayarlar =====
opts.quad_order     = 4;
opts.near_sing_eps  = 2.5;

opts.self_subdiv = 2;
opts.self_alpha  = 0.20;

opts.near_subdiv = 2;
opts.near_alpha  = 0.15;
opts.near_order  = 4;

%% ===== PRECOMPUTE (en kritik hızlandırma) =====
pre.tri = precompute_tri_geom(mesh);
[pre.qp,  pre.qw ] = tri_quad_rule(opts.quad_order);
[pre.qp2, pre.qw2] = tri_quad_rule(opts.near_order);

%% ===== Parallel pool: worker sayısını DÜŞÜR =====
p = gcp('nocreate');
if isempty(p)
    % çoğu sistemde 2-4 worker daha iyi (RAM/CPU thrash azalır)
    try
        parpool("local", min(4, feature('numcores')));
    catch
        parpool("local");
    end
end

meshC   = parallel.pool.Constant(mesh);
rwgC    = parallel.pool.Constant(rwg);
excC    = parallel.pool.Constant(exc);
optsC   = parallel.pool.Constant(opts);
wPatchC = parallel.pool.Constant(wPatch);
wGndC   = parallel.pool.Constant(wGnd);
preC    = parallel.pool.Constant(pre);

Zin   = complex(zeros(1,N));
Iport = complex(zeros(1,N));

parfor ii = 1:N
    f = freqs(ii);
    omega = 2*pi*f;
    k = omega * sqrt(mu0 * eps_eff);

    meshL   = meshC.Value;
    rwgL    = rwgC.Value;
    excL    = excC.Value;
    optsL   = optsC.Value;
    wPatchL = wPatchC.Value;
    wGndL   = wGndC.Value;
    preL    = preC.Value;

    [Z, Vrhs] = assemble_ZV_EFIE_RWG_DIFF_CPU_PRE(meshL, rwgL, k, omega, mu0, eps_eff, excL, optsL, preL);

    I = Z \ Vrhs;

    Ipatch = sum( (wPatchL(:) .* I(:)) .* rwgL.len(:) );
    Ignd   = sum( (wGndL(:)   .* I(:)) .* rwgL.len(:) );
    Iport(ii) = Ipatch - Ignd;

    Zin(ii) = excL.V0 / Iport(ii);
end

for ii = 1:N
    fprintf("(%2d/%2d) f=%.3f GHz | Zin = %.6f %+.6fj Ohm\n", ...
        ii, N, freqs(ii)/1e9, real(Zin(ii)), imag(Zin(ii)));
end

figure;
plot(freqs/1e9, real(Zin), '-o'); hold on;
plot(freqs/1e9, imag(Zin), '-s'); grid on;
xlabel('f (GHz)'); ylabel('Z_{in} (\Omega)');
legend('Re\{Z_{in}\}', 'Im\{Z_{in}\}', 'Location','best');
title('Giriş Empedansı Frekans Süpürmesi');
end
