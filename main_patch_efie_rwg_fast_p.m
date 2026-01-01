function main_patch_efie_rwg_fast_p(nWorkers)
% clear; 
% delete(gcp('nocreate'));


%% ===== Fizik =====
mu0  = 4*pi*1e-7;
c    = 299792458;
eps0 = 1/(mu0*c^2);

%% ===== Sweep =====
freqs = linspace(1.0e9, 1.4e9, 41);
N = numel(freqs);

%% ===== Substrat =====
er   = 2.59;
tand = 1e-4;

%% ===== Geometri =====
Lp = 20.1e-3;   Wp = 20.1e-3;
Lg = 0.20;      Wg = 0.20;
h  = 1.59e-3;

epsr_eff = (er+1)/2 + (er-1)/2 * (1 + 12*h/Wp)^(-0.5);
eps_eff  = eps0 * epsr_eff * (1 - 1j*tand);

%% ===== Mesh =====
Np_x = 8;  Np_y = 6;
Ng_x = 12;  Ng_y = 12;

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
opts.quad_order     = 2;
opts.near_sing_eps  = 1.2;

opts.self_subdiv = 2;
opts.self_alpha  = 0.20;

opts.near_subdiv = 1;
opts.near_alpha  = 0.15;
opts.near_order  = 2;

opts.use_symmetry = false; % önce false

%% ===== PRECOMPUTE (FAST'ın ana olayı) =====
tic;
pre = efie_precompute(mesh, opts);
fprintf("Precompute: %.2f s\n", toc);

%% ===== Sweep sonuçları =====
Zin   = complex(zeros(N,1));
Iport = complex(zeros(N,1));

% İstersen bunu true yapıp frekansları paralelleyebilirsin.
% Ama RAM thrash olursa daha yavaş olur (özellikle Ne büyükse).
usePar = true;

if nargin < 1 || isempty(nWorkers)
    nWorkers = 4; % default
end

if usePar && license('test','Distrib_Computing_Toolbox')
    p = gcp('nocreate');
    if isempty(p)
        parpool('local', nWorkers);
    elseif p.NumWorkers ~= nWorkers
        delete(p);
        parpool('local', nWorkers);
    end

    meshC   = parallel.pool.Constant(mesh);
    rwgC    = parallel.pool.Constant(rwg);
    excC    = parallel.pool.Constant(exc);
    optsC   = parallel.pool.Constant(opts);
    preC    = parallel.pool.Constant(pre);
    wPatchC = parallel.pool.Constant(wPatch);
    wGndC   = parallel.pool.Constant(wGnd);


    parfor ii = 1:N
        f = freqs(ii);
        omega = 2*pi*f;
        k = omega * sqrt(mu0 * eps_eff);

        [Z, Vrhs] = assemble_ZV_EFIE_RWG_DIFF_CPU_FAST( ...
            meshC.Value, rwgC.Value, k, omega, mu0, eps_eff, excC.Value, optsC.Value, preC.Value);

        I = Z \ Vrhs;

        Ipatch = sum( (wPatchC.Value(:).*I(:)).*rwgC.Value.len(:) );
        Ignd   = sum( (wGndC.Value(:)  .*I(:)).*rwgC.Value.len(:) );
        Iport(ii) = Ipatch - Ignd;

        Zin(ii) = excC.Value.V0 / Iport(ii);
    end
else
    for ii = 1:N
        f = freqs(ii);
        omega = 2*pi*f;
        k = omega * sqrt(mu0 * eps_eff);

        fprintf("(%2d/%2d) f=%.3f GHz\n", ii, N, f/1e9);

        [Z, Vrhs] = assemble_ZV_EFIE_RWG_DIFF_CPU_FAST(mesh, rwg, k, omega, mu0, eps_eff, exc, opts, pre);
        I = Z \ Vrhs;

        Ipatch = sum( (wPatch(:).*I(:)).*rwg.len(:) );
        Ignd   = sum( (wGnd(:)  .*I(:)).*rwg.len(:) );
        Iport(ii) = Ipatch - Ignd;

        Zin(ii) = exc.V0 / Iport(ii);

        fprintf("    Zin = %.6f %+.6fj Ohm\n", real(Zin(ii)), imag(Zin(ii)));
    end
end

%% ===== Plot =====
figure;
plot(freqs/1e9, real(Zin), '-o'); hold on;
plot(freqs/1e9, imag(Zin), '-s'); grid on;
xlabel('f (GHz)'); ylabel('Z_{in} (\Omega)');
legend('Re\{Z_{in}\}', 'Im\{Z_{in}\}', 'Location','best');
title('Giriş Empedansı (FAST)');

end
