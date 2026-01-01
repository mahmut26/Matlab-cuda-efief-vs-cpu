function prepack = efie_precompute_pack(pre, opts)
% pre: efie_precompute(...) çıktısı
% opts: aynı opts (self_subdiv, near_subdiv, quad_order, near_order, ...)

tri = pre.tri;
Nt  = size(tri.r1,1);

% ---- regular quad ----
r_reg = pre.r_reg;   % [Nq x 3 x Nt]
w_reg = pre.w_reg;   % [Nq x Nt]  (qw*area2) içeriyor

% ---- nearMat ----
nearMat = uint8(pre.nearMat); % [Nt x Nt]

% ---- self cache (cell -> packed) ----
% NsS = 4^self_subdiv
NsS = size(pre.self{1}.r,3);
Nq2 = size(pre.self{1}.r,1);

self_r = zeros(Nq2,3,NsS,Nt,'double');
self_w = zeros(Nq2,NsS,Nt,'double');
self_a = zeros(NsS,Nt,'double');

for t=1:Nt
    self_r(:,:,:,t) = pre.self{t}.r;
    self_w(:,:,t)   = pre.self{t}.w;
    self_a(:,t)     = pre.self{t}.aSoft(:);
end

% ---- near cache (cell -> packed) ----
NsN = size(pre.near{1}.r,3);
near_r = zeros(Nq2,3,NsN,Nt,'double');
near_w = zeros(Nq2,NsN,Nt,'double');

for t=1:Nt
    near_r(:,:,:,t) = pre.near{t}.r;
    near_w(:,:,t)   = pre.near{t}.w;
end

% ---- aeqTri ----
aeqTri = pre.aeqTri(:);  % [Nt x 1]

prepack.r_reg   = r_reg;
prepack.w_reg   = w_reg;
prepack.nearMat = nearMat;

prepack.self_r = self_r;
prepack.self_w = self_w;
prepack.self_a = self_a;

prepack.near_r = near_r;
prepack.near_w = near_w;

prepack.aeqTri = aeqTri;

% opts’ların GPU tarafında lazım olanlarını da koy
prepack.near_alpha = opts.near_alpha;
prepack.self_alpha = opts.self_alpha; %#ok<NASGU> (MEX’e direkt vermiyorsan şart değil)
end
