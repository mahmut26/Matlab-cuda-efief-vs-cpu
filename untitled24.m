workerList = 2:6;
nRuns = 4;          % warmup hariç ekstra 5
nTotal = nRuns + 1; % warmup dahil toplam

T = zeros(numel(workerList), nTotal);

% Figürleri gizle (fonksiyon içindeki figure açmaları süreyi bozmasın)
oldVis = get(0,'DefaultFigureVisible');
set(0,'DefaultFigureVisible','off');
cleanup = onCleanup(@() set(0,'DefaultFigureVisible', oldVis));

for wIdx = 1:numel(workerList)
    nw = workerList(wIdx);

    % Pool'u nw işçiyle aç / yeniden ayarla (bu kısım ölçüme dahil değil)
    p = gcp('nocreate');
    if isempty(p)
        p = parpool('local', nw);
    elseif p.NumWorkers ~= nw
        delete(p);
        p = parpool('local', nw);
    end
    p.IdleTimeout = Inf;

    fprintf('\n=== nw=%d | toplam %d run (ilk dahil) ===\n', nw, nTotal);

    for r = 1:nTotal
        t0 = tic;
        main_patch_efie_rwg_fast_p(nw);
        T(wIdx, r) = toc(t0);

        fprintf('nw=%d | run=%d/%d | %.3f s\n', nw, r, nTotal, T(wIdx,r));
        drawnow;
    end
end

% Tablo göster
disp(array2table(T, ...
    'RowNames', compose("nw=%d", workerList), ...
    'VariableNames', compose("run%d", 1:nTotal)));

% Özet (ilk dahil)
summary = table(workerList(:), mean(T,2), std(T,0,2), min(T,[],2), max(T,[],2), ...
    'VariableNames', {'Workers','Mean_s','Std_s','Min_s','Max_s'});
disp(summary);

nRuns = 5;                 % kaç kere ölçülecek
T = zeros(1, nRuns);

% Figürleri gizle (fonksiyon içindeki figure açmaları süreyi bozmasın)
oldVis = get(0,'DefaultFigureVisible');
set(0,'DefaultFigureVisible','off');
cleanup = onCleanup(@() set(0,'DefaultFigureVisible', oldVis));

% Tek CPU koşu için pool kapalı olsun
delete(gcp('nocreate'));

fprintf('\n=== main_patch_efie_rwg_fast() | toplam %d run ===\n', nRuns);

for r = 1:nRuns
    t0 = tic;
    main_patch_efie_rwg_fast();      % <-- argümansız
    T(r) = toc(t0);

    fprintf('run=%d/%d | %.3f s\n', r, nRuns, T(r));
    drawnow;
end

% Tablo göster
disp(array2table(T, 'VariableNames', compose("run%d", 1:nRuns)));

% Özet
summary = table(mean(T), std(T,0,2), min(T), max(T), ...
    'VariableNames', {'Mean_s','Std_s','Min_s','Max_s'});
disp(summary);



nRuns = 5;                 % kaç kere ölçülecek
T = zeros(1, nRuns);

% Figürleri gizle (fonksiyon içindeki figure açmaları süreyi bozmasın)
oldVis = get(0,'DefaultFigureVisible');
set(0,'DefaultFigureVisible','off');
cleanup = onCleanup(@() set(0,'DefaultFigureVisible', oldVis));

% Tek CPU koşu için pool kapalı olsun
delete(gcp('nocreate'));

fprintf('\n=== main_patch_efie_rwg_fast() | toplam %d run ===\n', nRuns);

for r = 1:nRuns
    t0 = tic;
    main_patch_gpu_mex_sweep_l();      % <-- argümansız
    T(r) = toc(t0);

    fprintf('run=%d/%d | %.3f s\n', r, nRuns, T(r));
    drawnow;
end

% Tablo göster
disp(array2table(T, 'VariableNames', compose("run%d", 1:nRuns)));

% Özet
summary = table(mean(T), std(T,0,2), min(T), max(T), ...
    'VariableNames', {'Mean_s','Std_s','Min_s','Max_s'});
disp(summary);
