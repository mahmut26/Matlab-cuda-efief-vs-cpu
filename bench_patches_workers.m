function [times, avgGHz, summary] = bench_patches_workers()

workerList = 2:6;

names = { ...
    'main_patch_efie_rwg_fast_p', ...
    'main_patch_efie_rwg_fast', ...
    'main_patch_gpu_mex_sweep_l' ...
};

nWarmup = 0;
nRuns   = 5;

sample_dt = 1.0;   % CPU frekansı örnekleme aralığı (s)

nW  = numel(workerList);
nFn = numel(names);

times  = zeros(nW, nFn, nRuns);
avgGHz = NaN(nW, nFn, nRuns);

for wIdx = 1:nW
    nw = workerList(wIdx);

    % Havuzu bu worker sayısıyla aç
    p = gcp('nocreate');
    if isempty(p)
        p = parpool('local', nw);
    elseif p.NumWorkers ~= nw
        delete(p);
        p = parpool('local', nw);
    end
    p.IdleTimeout = Inf;  % uzun koşularda pool kapanmasın

    % Fonksiyon handle’larını worker sayısına bağla
    fns = { ...
        @() main_patch_efie_rwg_fast_p(nw), ...
        @() main_patch_efie_rwg_fast(nw), ...
        @() main_patch_gpu_mex_sweep_l(nw) ...
    };

    % Warmup (istersen)
    for i = 1:nFn
        for k = 1:nWarmup
            fns{i}();
        end
    end

    % Ölçüm
    for i = 1:nFn
        for r = 1:nRuns

            if parallel.gpu.GPUDevice.isAvailable
                wait(gpuDevice);
            end

            % === Frekans örnekleme (BU RUN İÇİN) ===
            freq_samples_mhz = [];  % <-- HER RUN BAŞINDA RESET

            tmr = timer( ...
                'ExecutionMode','fixedRate', ...
                'Period', sample_dt, ...
                'BusyMode','drop', ...
                'TimerFcn', @sample_freq);

            % Timer kapanışını garanti et
            c = onCleanup(@() safe_stop_delete(tmr));

            mhz0 = cpu_mhz();  % fallback
            start(tmr);

            t0 = tic;
            fns{i}();

            if parallel.gpu.GPUDevice.isAvailable
                wait(gpuDevice);
            end

            elapsed = toc(t0);

            % Timer'ı kapat (onCleanup zaten garanti ediyor)
            safe_stop_delete(tmr);

            mhz1 = cpu_mhz();

            if ~isempty(freq_samples_mhz)
                mhz_avg = mean(freq_samples_mhz, 'omitnan');
            else
                mhz_avg = mean([mhz0 mhz1], 'omitnan');
            end

            times(wIdx, i, r)  = elapsed;
            avgGHz(wIdx, i, r) = mhz_avg / 1000;

            fprintf('[%d workers] %s | run %d/%d : %.3f s | avgCPU=%.2f GHz\n', ...
                nw, names{i}, r, nRuns, times(wIdx,i,r), avgGHz(wIdx,i,r));
            drawnow;

            clear c; % cleanup handle
        end
    end
end

% === Özet tablo (time + freq) ===
rows = nW * nFn;
Workers = zeros(rows,1);
Func    = strings(rows,1);

Mean_s = zeros(rows,1);
Std_s  = zeros(rows,1);
Min_s  = zeros(rows,1);
Max_s  = zeros(rows,1);

Mean_GHz = NaN(rows,1);
Std_GHz  = NaN(rows,1);
Min_GHz  = NaN(rows,1);
Max_GHz  = NaN(rows,1);

idx = 0;
for wIdx = 1:nW
    for i = 1:nFn
        idx = idx + 1;

        t = squeeze(times(wIdx,i,:));
        g = squeeze(avgGHz(wIdx,i,:));

        Workers(idx) = workerList(wIdx);
        Func(idx)    = names{i};

        Mean_s(idx) = mean(t);
        Std_s(idx)  = std(t);
        Min_s(idx)  = min(t);
        Max_s(idx)  = max(t);

        Mean_GHz(idx) = mean(g, 'omitnan');
        Std_GHz(idx)  = std(g,  'omitnan');
        Min_GHz(idx)  = min(g, [], 'omitnan');
        Max_GHz(idx)  = max(g, [], 'omitnan');
    end
end

summary = table(Workers, Func, Mean_s, Std_s, Min_s, Max_s, ...
                Mean_GHz, Std_GHz, Min_GHz, Max_GHz);

disp(summary);

% ===== nested helpers =====
    function sample_freq(~,~)
        freq_samples_mhz(end+1,1) = cpu_mhz(); %#ok<AGROW>
    end

    function safe_stop_delete(t)
        if isa(t,'timer') && isvalid(t)
            try, stop(t);  catch, end
            try, delete(t); catch, end
        end
    end

    function mhz = cpu_mhz()
        mhz = NaN;

        if ispc
            [~,out] = system('wmic cpu get CurrentClockSpeed /value');
            tok = regexp(out,'CurrentClockSpeed=(\d+)','tokens','once');
            if ~isempty(tok), mhz = str2double(tok{1}); end

        elseif isunix && ~ismac
            [st,out] = system('cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq');
            if st==0
                mhz = str2double(strtrim(out))/1000;
            else
                [~,out] = system("lscpu | awk -F: '/CPU MHz/ {gsub(/ /,"""",$2); print $2; exit}'");
                v = str2double(strtrim(out));
                if ~isnan(v), mhz = v; end
            end
        end
    end
end
