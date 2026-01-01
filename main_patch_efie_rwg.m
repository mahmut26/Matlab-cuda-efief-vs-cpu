% % function main_patch_efie_rwg()
% % clc; clear; close all;
% % 
% % %% ===== Fizik =====
% % c   = 299792458;
% % mu0 = 4*pi*1e-7;
% % eps0 = 1/(mu0*c^2);
% % 
% % f = 0.48e9;
% % omega = 2*pi*f;
% % 
% % %% ===== Substrat/etkin epsilon =====
% % er   = 2.59;
% % tand = 1e-4;
% % 
% % %% ===== Geometri =====
% % Lp = 20.1e-3;   % y (length)
% % Wp = 20.1e-3;   % x (width)
% % Lg = 0.20;      % y
% % Wg = 0.20;      % x
% % h  = 1.59e-3;   % patch-ground separation
% % 
% % % microstrip epsr_eff
% % epsr_eff = (er+1)/2 + (er-1)/2 * (1 + 12*h/Wp)^(-0.5);
% % eps_eff  = eps0 * epsr_eff * (1 - 1j*tand);
% % 
% % % tutarlı k:
% % k = omega * sqrt(mu0 * eps_eff);
% % 
% % fprintf("f=%.3f GHz | epsr_eff=%.3f | lambda_eff=%.4f m\n", ...
% %     f/1e9, real(epsr_eff), 2*pi/real(k));
% % 
% % %% ===== Mesh çözünürlüğü =====
% % Np_x = 12;  Np_y = 8;    % patch
% % Ng_x = 18;  Ng_y = 18;   % ground
% % 
% % [V,F,patchFaceIdx,groundFaceIdx] = make_patch_with_ground_mesh( ...
% %     Lp, Wp, Lg, Wg, h, Np_x, Np_y, Ng_x, Ng_y);
% % 
% % mesh.V = V;
% % mesh.F = F;
% % 
% % fprintf("Vertices=%d | Triangles=%d\n", size(V,1), size(F,1));
% % 
% % %% ===== RWG =====
% % rwg = build_rwg_interior(mesh);   % interior-only, orientation-aware
% % fprintf("RWG unknowns=%d\n", rwg.Ne);
% % 
% % %% ===== Feed noktası =====
% % insetDepth = 13.2e-3;
% % 
% % feed_x = -Wp/2 + 1e-3;
% % feed_y = -Lp/2 + insetDepth;
% % 
% % portPointPatch = [feed_x, feed_y, +h/2];
% % portPointGnd   = [feed_x, feed_y, -h/2];
% % 
% % portEdgePatch = pick_port_edge_by_point_and_layer(rwg, portPointPatch, +1);
% % portEdgeGnd   = pick_port_edge_by_point_and_layer(rwg, portPointGnd,   -1);
% % 
% % fprintf("portEdgePatch=%d | portEdgeGnd=%d\n", portEdgePatch, portEdgeGnd);
% % 
% % %% ===== Uyarım (diferansiyel edge-strip) =====
% % exc.V0 = 1.0;
% % exc.patchEdge  = portEdgePatch;
% % exc.groundEdge = portEdgeGnd;
% % exc.spreadRadius = 1.5 * mean(rwg.len);
% % 
% % %% ===== Numerik ayarlar =====
% % opts.quad_order     = 4;
% % opts.near_sing_eps  = 2.5;
% % 
% % opts.self_subdiv = 2;
% % opts.self_alpha  = 0.20;
% % 
% % opts.near_subdiv = 2;
% % opts.near_alpha  = 0.15;
% % opts.near_order  = 4;
% % 
% % %% ===== Z ve RHS =====
% % [Z, Vrhs] = assemble_ZV_EFIE_RWG_DIFF_CPU(mesh, rwg, k, omega, mu0, eps_eff, exc, opts);
% % 
% % %% ===== Çöz =====
% % I = Z \ Vrhs;   % <-- EKSİK OLAN SATIR buydu
% % 
% % % Port ağırlıkları (RHS ile aynı mantık)
% % [wPatch, wGnd] = port_weights_gaussian(mesh, rwg, exc.patchEdge, exc.groundEdge, exc.spreadRadius);
% % 
% % % “Toplanmış” port akımı (PoC)
% % Ipatch = sum( (wPatch(:) .* I(:)) .* rwg.len(:) );
% % Ignd   = sum( (wGnd(:)   .* I(:)) .* rwg.len(:) );
% % Iport  = Ipatch - Ignd;
% % 
% % Zin = exc.V0 / Iport;
% % 
% % fprintf("Zin = %.6f %+.6fj Ohm\n", real(Zin), imag(Zin));
% % end
% function main_patch_efie_rwg()
% clc; clear; close all;
% 
% %% ===== Fizik =====
% c   = 299792458;
% mu0 = 4*pi*1e-7;
% eps0 = 1/(mu0*c^2);
% 
% %% ===== Frekans süpürmesi =====
% freqs = linspace(1.0e9, 1.4e9, 41);
% 
% %% ===== Substrat/etkin epsilon =====
% er   = 2.59;
% tand = 1e-4;
% 
% %% ===== Geometri =====
% Lp = 20.1e-3;   % y (length)
% Wp = 20.1e-3;   % x (width)
% Lg = 0.20;      % y
% Wg = 0.20;      % x
% h  = 1.59e-3;   % patch-ground separation
% 
% % microstrip epsr_eff (f'den bağımsız varsayıldı)
% epsr_eff = (er+1)/2 + (er-1)/2 * (1 + 12*h/Wp)^(-0.5);
% eps_eff  = eps0 * epsr_eff * (1 - 1j*tand);
% 
% %% ===== Mesh çözünürlüğü =====
% Np_x = 12;  Np_y = 8;    % patch
% Ng_x = 18;  Ng_y = 18;   % ground
% 
% [V,F,patchFaceIdx,groundFaceIdx] = make_patch_with_ground_mesh( ...
%     Lp, Wp, Lg, Wg, h, Np_x, Np_y, Ng_x, Ng_y);
% 
% mesh.V = V;
% mesh.F = F;
% 
% fprintf("Vertices=%d | Triangles=%d\n", size(V,1), size(F,1));
% 
% %% ===== RWG =====
% rwg = build_rwg_interior(mesh);   % interior-only, orientation-aware
% fprintf("RWG unknowns=%d\n", rwg.Ne);
% 
% %% ===== Feed noktası =====
% insetDepth = 13.2e-3;
% 
% feed_x = -Wp/2 + 1e-3;
% feed_y = -Lp/2 + insetDepth;
% 
% portPointPatch = [feed_x, feed_y, +h/2];
% portPointGnd   = [feed_x, feed_y, -h/2];
% 
% portEdgePatch = pick_port_edge_by_point_and_layer(rwg, portPointPatch, +1);
% portEdgeGnd   = pick_port_edge_by_point_and_layer(rwg, portPointGnd,   -1);
% 
% fprintf("portEdgePatch=%d | portEdgeGnd=%d\n", portEdgePatch, portEdgeGnd);
% 
% %% ===== Uyarım (diferansiyel edge-strip) =====
% exc.V0 = 1.0;
% exc.patchEdge  = portEdgePatch;
% exc.groundEdge = portEdgeGnd;
% exc.spreadRadius = 1.5 * mean(rwg.len);
% 
% % Port ağırlıkları (f'den bağımsız -> bir kere hesapla)
% [wPatch, wGnd] = port_weights_gaussian(mesh, rwg, exc.patchEdge, exc.groundEdge, exc.spreadRadius);
% 
% %% ===== Numerik ayarlar =====
% opts.quad_order     = 4;
% opts.near_sing_eps  = 2.5;
% 
% opts.self_subdiv = 2;
% opts.self_alpha  = 0.20;
% 
% opts.near_subdiv = 2;
% opts.near_alpha  = 0.15;
% opts.near_order  = 4;
% 
% %% ===== Sweep çıktıları =====
% Zin   = complex(zeros(size(freqs)));
% Iport = complex(zeros(size(freqs)));
% 
% %% ===== Frekans döngüsü =====
% for ii = 1:numel(freqs)
%     f = freqs(ii);
%     omega = 2*pi*f;
% 
%     % tutarlı k:
%     k = omega * sqrt(mu0 * eps_eff);
% 
%     fprintf("(%2d/%2d) f=%.3f GHz | epsr_eff=%.3f | lambda_eff=%.4f m\n", ...
%         ii, numel(freqs), f/1e9, real(epsr_eff), 2*pi/real(k));
% 
%     %% Z ve RHS
%     [Z, Vrhs] = assemble_ZV_EFIE_RWG_DIFF_CPU(mesh, rwg, k, omega, mu0, eps_eff, exc, opts);
% 
%     %% Çöz
%     I = Z \ Vrhs;
% 
%     %% Port akımı ve giriş empedansı
%     Ipatch = sum( (wPatch(:) .* I(:)) .* rwg.len(:) );
%     Ignd   = sum( (wGnd(:)   .* I(:)) .* rwg.len(:) );
%     Iport(ii) = Ipatch - Ignd;
% 
%     Zin(ii) = exc.V0 / Iport(ii);
% 
%     fprintf("    Zin = %.6f %+.6fj Ohm\n", real(Zin(ii)), imag(Zin(ii)));
% end
% 
% %% ===== Basit plot =====
% figure;
% plot(freqs/1e9, real(Zin), '-o'); hold on;
% plot(freqs/1e9, imag(Zin), '-s'); grid on;
% xlabel('f (GHz)'); ylabel('Z_{in} (\Omega)');
% legend('Re\{Z_{in}\}', 'Im\{Z_{in}\}', 'Location','best');
% title('Giriş Empedansı Frekans Süpürmesi');
% 
% end
function main_patch_efie_rwg()
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
Lp = 20.1e-3;   % y (length)
Wp = 20.1e-3;   % x (width)
Lg = 0.20;      % y
Wg = 0.20;      % x
h  = 1.59e-3;   % patch-ground separation

% microstrip epsr_eff (f'den bağımsız varsayıldı)
epsr_eff = (er+1)/2 + (er-1)/2 * (1 + 12*h/Wp)^(-0.5);
eps_eff  = eps0 * epsr_eff * (1 - 1j*tand);

%% ===== Mesh çözünürlüğü =====
Np_x = 12;  Np_y = 8;    % patch
Ng_x = 18;  Ng_y = 18;   % ground

[V,F,patchFaceIdx,groundFaceIdx] = make_patch_with_ground_mesh( ...
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

% Port ağırlıkları (f'den bağımsız -> bir kere hesapla)
[wPatch, wGnd] = port_weights_gaussian(mesh, rwg, exc.patchEdge, exc.groundEdge, exc.spreadRadius);

%% ===== Numerik ayarlar =====
opts.quad_order     = 4;
opts.near_sing_eps  = 2.5;

opts.self_subdiv = 2;
opts.self_alpha  = 0.20;

opts.near_subdiv = 2;
opts.near_alpha  = 0.15;
opts.near_order  = 4;

%% ===== Parallel pool (varsa) =====
if isempty(gcp('nocreate'))
    % threads çoğu zaman mesh/rwg broadcast overhead'ini azaltır
    try
        parpool("threads");
    catch
        parpool("local");
    end
end

% Büyük objeleri worker başına 1 kere sabitle (broadcast maliyetini azaltır)
meshC   = parallel.pool.Constant(mesh);
rwgC    = parallel.pool.Constant(rwg);
excC    = parallel.pool.Constant(exc);
optsC   = parallel.pool.Constant(opts);
wPatchC = parallel.pool.Constant(wPatch);
wGndC   = parallel.pool.Constant(wGnd);

%% ===== Sweep çıktıları =====
Zin   = complex(zeros(1,N));
Iport = complex(zeros(1,N));

%% ===== Frekans döngüsü (PARFOR) =====
parfor ii = 1:N
    f = freqs(ii);
    omega = 2*pi*f;

    % tutarlı k:
    k = omega * sqrt(mu0 * eps_eff);

    % worker-local kopyalar
    meshL   = meshC.Value;
    rwgL    = rwgC.Value;
    excL    = excC.Value;
    optsL   = optsC.Value;
    wPatchL = wPatchC.Value;
    wGndL   = wGndC.Value;

    % Z ve RHS
    [Z, Vrhs] = assemble_ZV_EFIE_RWG_DIFF_CPU(meshL, rwgL, k, omega, mu0, eps_eff, excL, optsL);

    % Çöz
    I = Z \ Vrhs;

    % Port akımı ve giriş empedansı
    Ipatch = sum( (wPatchL(:) .* I(:)) .* rwgL.len(:) );
    Ignd   = sum( (wGndL(:)   .* I(:)) .* rwgL.len(:) );
    Iport(ii) = Ipatch - Ignd;

    Zin(ii) = excL.V0 / Iport(ii);
end

%% ===== Sonuç yazdırma (seri) =====
for ii = 1:N
    fprintf("(%2d/%2d) f=%.3f GHz | Zin = %.6f %+.6fj Ohm\n", ...
        ii, N, freqs(ii)/1e9, real(Zin(ii)), imag(Zin(ii)));
end

%% ===== Basit plot =====
figure;
plot(freqs/1e9, real(Zin), '-o'); hold on;
plot(freqs/1e9, imag(Zin), '-s'); grid on;
xlabel('f (GHz)'); ylabel('Z_{in} (\Omega)');
legend('Re\{Z_{in}\}', 'Im\{Z_{in}\}', 'Location','best');
title('Giriş Empedansı Frekans Süpürmesi');

end
