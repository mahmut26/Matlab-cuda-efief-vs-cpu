function [times, summary] = bench_patches_workers()

workerList = 2:6;

names = { ...
    'main_patch_efie_rwg_fast_p', ...
    'main_patch_efie_rwg_fast', ...
    'main_patch_gpu_mex_sweep_l' ...
};

nWarmup = 0;   % uzun işler için warmup genelde gereksiz/pahalı
nRuns   = 5;

nW  = numel(workerList);
nFn = numel(names);

times = zeros(nW, nFn, nRuns);

for wIdx = 1:nW
    nw = workerList(wIdx);

    % Havuzu bu worker sayısıyla aç
    p = gcp('nocreate');
    if isempty(p)
        parpool('local', nw);
    elseif p.NumWorkers ~= nw
        delete(p);
        parpool('local', nw);
    end

    % Fonksiyon handle’larını bu worker sayısına bağla
    fns = { ...
        @() main_patch_efie_rwg_fast_p(nw), ...
        @() main_patch_efie_rwg_fast(nw), ...
        @() main_patch_gpu_mex_sweep_l(nw) ...
    };

    % Warmup (istersen)
    for i = 1:nFn
        for k = 1:nWarmup
            fns{i}();
            if parallel.gpu.GPUDevice.isAvailable
                wait(gpuDevice);
            end
        end
    end

    % Ölçüm
    for i = 1:nFn
        for r = 1:nRuns
            if parallel.gpu.GPUDevice.isAvailable
                wait(gpuDevice);
            end

            t0 = tic;
            fns{i}();

            if parallel.gpu.GPUDevice.isAvailable
                wait(gpuDevice);
            end

            times(wIdx, i, r) = toc(t0);
            fprintf('[%d workers] %s | run %d/%d : %.3f s\n', ...
                nw, names{i}, r, nRuns, times(wIdx,i,r));
            drawnow;
        end
    end
end

% Özet tablo (mean/std) — her worker × her fonksiyon
rows = nW * nFn;
Wcol = zeros(rows,1);
Fcol = strings(rows,1);
Mean = zeros(rows,1);
Std  = zeros(rows,1);
MinT = zeros(rows,1);
MaxT = zeros(rows,1);

idx = 0;
for wIdx = 1:nW
    for i = 1:nFn
        idx = idx + 1;
        t = squeeze(times(wIdx,i,:));
        Wcol(idx) = workerList(wIdx);
        Fcol(idx) = names{i};
        Mean(idx) = mean(t);
        Std(idx)  = std(t);
        MinT(idx) = min(t);
        MaxT(idx) = max(t);
    end
end

summary = table(Wcol, Fcol, Mean, Std, MinT, MaxT, ...
    'VariableNames', {'Workers','Function','Mean_s','Std_s','Min_s','Max_s'});

disp(summary)
end
