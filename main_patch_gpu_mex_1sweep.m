function main_patch_gpu_mex_1sweep()
clc; clear; 

%% ===== Fizik =====
mu0  = 4*pi*1e-7;
c    = 299792458;
eps0 = 1/(mu0*c^2);

%% ===== Sweep =====
freqs = linspace(1.0e9, 1.4e9, 41).';   % Nf x 1
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

fprintf("epsr_eff=%.6f, tand=%.2g\n", real(epsr_eff), tand);

%% ===== Mesh =====
Np_x = 8;  Np_y = 6;
Ng_x = 12; Ng_y = 12;

[V,F] = make_patch_with_ground_mesh(Lp, Wp, Lg, Wg, h, Np_x, Np_y, Ng_x, Ng_y);
mesh.V = V;
mesh.F = int32(F);   % MEX: int32 bekliyor

fprintf("Vertices=%d | Triangles=%d\n", size(mesh.V,1), size(mesh.F,1));

%% ===== RWG =====
rwg = build_rwg_interior(struct('V',mesh.V,'F',double(mesh.F))); % RWG builder double F isteyebilir
fprintf("RWG unknowns=%d\n", rwg.Ne);

% --- MEX için cast ---
rwg.edge      = int32(rwg.edge);
rwg.plusTri   = int32(rwg.plusTri);
rwg.minusTri  = int32(rwg.minusTri);
rwg.plusSign  = int32(rwg.plusSign);
rwg.minusSign = int32(rwg.minusSign);

%% ===== Feed =====
insetDepth = 13.2e-3;
feed_x = -Wp/2 + 1e-3;
feed_y = -Lp/2 + insetDepth;

portPointPatch = [feed_x, feed_y, +h/2];
portPointGnd   = [feed_x, feed_y, -h/2];

portEdgePatch = pick_port_edge_by_point_and_layer(rwg, portPointPatch, +1);
portEdgeGnd   = pick_port_edge_by_point_and_layer(rwg, portPointGnd,   -1);

fprintf("portEdgePatch=%d | portEdgeGnd=%d\n", portEdgePatch, portEdgeGnd);

%% ===== Uyarım parametreleri (MEX’in kullandığı) =====
V0 = 1.0;
spreadRadius = 1.5 * mean(double(rwg.len));  % MEX RHS gaussian için
Z0 = 50;

%% ===== MEX var mı? =====
% Senin CUDA kodunun mex adı neyse buraya onu yaz.
% Örnek: patch_sweep_cuda veya assemble_ZV_EFIE_RWG_DIFF_GPU_FAST
mexName = '';
if exist('patch_sweep_cuda','file') == 3
    mexName = 'patch_sweep_cuda';
elseif exist('assemble_ZV_EFIE_RWG_DIFF_GPU_FAST','file') == 3
    mexName = 'assemble_ZV_EFIE_RWG_DIFF_GPU_FAST';
else
    error("GPU MEX bulunamadı. Derledikten sonra mex dosyası path üzerinde olmalı.");
end

%% ===== Sweep =====
Zin = complex(zeros(N,1));
S11 = complex(zeros(N,1));

tic;
switch mexName
    case 'patch_sweep_cuda'
        % Bu imza: [Zin,S11] = patch_sweep_cuda(..., freqs, mu0, eps_eff, patchEdge, groundEdge, V0, spreadRadius, Z0)
        [Zin, S11] = patch_sweep_cuda( ...
            mesh.V, mesh.F, rwg.edge, ...
            rwg.plusTri, rwg.minusTri, ...
            rwg.plusSign, rwg.minusSign, ...
            rwg.rp, rwg.rm, rwg.Ap, rwg.Am, double(rwg.len), ...
            freqs, mu0, eps_eff, ...
            int32(portEdgePatch), int32(portEdgeGnd), ...
            V0, spreadRadius, Z0 );

    case 'assemble_ZV_EFIE_RWG_DIFF_GPU_FAST'
        % Eğer senin mex’in bu isimdeyse, genelde iki çıkış: [Zin,S11] döndürecek şekilde yazmıştın.
        % Eğer sen farklı imza kullandıysan, burayı ona göre güncelle.
        [Zin, S11] = assemble_ZV_EFIE_RWG_DIFF_GPU_FAST( ...
            mesh.V, mesh.F, rwg.edge, ...
            rwg.plusTri, rwg.minusTri, ...
            rwg.plusSign, rwg.minusSign, ...
            rwg.rp, rwg.rm, rwg.Ap, rwg.Am, double(rwg.len), ...
            freqs, mu0, eps_eff, ...
            int32(portEdgePatch), int32(portEdgeGnd), ...
            V0, spreadRadius, Z0 );
end
tGPU = toc;

fprintf("GPU sweep finished in %.2f s (Nf=%d)\n", tGPU, N);

%% ===== Plot =====
figure('Color','w');
subplot(2,1,1);
plot(freqs/1e9, real(Zin), '-o', 'LineWidth', 1.5); hold on;
plot(freqs/1e9, imag(Zin), '-s', 'LineWidth', 1.5);
grid on; xlabel('f (GHz)'); ylabel('Z_{in} (\Omega)');
legend('Re\{Zin\}','Im\{Zin\}','Location','best');
title('Zin (GPU MEX)');

subplot(2,1,2);
plot(freqs/1e9, 20*log10(abs(S11)), 'LineWidth', 1.5);
grid on; xlabel('f (GHz)'); ylabel('|S11| (dB)');
title('|S11| (GPU MEX)');

end
