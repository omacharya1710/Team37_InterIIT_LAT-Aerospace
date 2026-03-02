%% ========================================================================
%  parametric_AR_taper_iw_CL_CD_LD.m
%  Parametric study + CL, CD, L/D contour plots + assumption summary
%
%  - AR sweep:        8.5 : 0.1 : 12.0
%  - Taper ratio λ:   0.40 : 0.05 : 0.45  (c_tip / c_root)
%  - Wing setting i_w: fixed at 3 deg
%
%  Wing area S = 0.117 m^2, twist = 0, sweep = 0, N = 25 Fourier terms
%  Airfoils: NACA4412 + S1223 (2D slopes and alpha0 assumed as below)
%
%  Outputs:
%   - T: table with AR, lambda, i_w_deg, CL_wing, CDi, L_over_D  (CSV)
%   - Colour contour plots for CL, CD, L/D vs AR & λ
% ========================================================================

clc; clear; close all;

%% ---------------- ASSUMPTIONS / TWEAKS (EDIT HERE) ----------------
S = 0.117;                % wing area (m^2)
twist_deg = 0;            % total geometric twist root->tip (deg)
sweep_deg = 0;            % (not used explicitly in this lifting-line model)
N = 25;                   % number of Fourier terms / control points

% Parameter ranges
AR_vec     = 8.5 : 0.1 : 12.0;      % aspect ratio sweep
lambda_vec = 0.40 : 0.05 : 0.45;    % taper ratio sweep (lambda = c_tip/c_root)
i_w_deg    = 3.0;                   % single wing setting angle in degrees

% Airfoil 2D properties (assumed / baseline)
% NACA4412 main element
a2d_4412         = 6.3;    % 1/rad (approx; thin-airfoil ~ 2*pi = 6.283)
alpha0_4412_deg  = -1.5;   % zero-lift AoA (deg)

% S1223 second element (high-lift low-Re airfoil)
a2d_s1223        = 6.0;    % 1/rad (example)
alpha0_s1223_deg = -2.0;   % zero-lift AoA (deg)

a2d_vec        = [a2d_4412,        a2d_s1223];
alpha0_vec_deg = [alpha0_4412_deg, alpha0_s1223_deg];

%% ---------------- PREPARE STORAGE ----------------
nAR  = numel(AR_vec);
nLam = numel(lambda_vec);

CL_results      = nan(nAR, nLam);   % total CL for each (AR, lambda)
CDi_results     = nan(nAR, nLam);   % induced CD
LoverD_results  = nan(nAR, nLam);   % L/D
A1_results      = nan(nAR, nLam);   % first Fourier coef (optional)

%% ---------------- MAIN LOOPS ----------------
for ia = 1:nAR
    AR = AR_vec(ia);
    for il = 1:nLam
        lambda = lambda_vec(il);

        % Geometry that depends on AR & lambda
        b   = sqrt(AR * S);        % span (m)
        MAC = S / b;               % mean aerodynamic chord
        Croot = (1.5*(1+lambda)*MAC)/(1+lambda+lambda^2);
        Ctip  = lambda * Croot;

        % Control point angles and positions
        theta   = (pi/(2*N)) : (pi/(2*N)) : pi/2;   % N control pts
        c_theta = Croot * (1 - (1 - lambda) * cos(theta));  % chord at each control pt

        % Precompute sine basis
        J = 1:N;
        K = 2*J - 1;                           % harmonic indices: 1,3,5,...
        [ThetaGrid, Kgrid] = ndgrid(theta, K);
        S_basis = sin(Kgrid .* ThetaGrid);     % N x N

        % AoA distribution (no twist here except constant i_w)
        alpha_geom_deg = linspace(i_w_deg + twist_deg, i_w_deg, N); % root->tip
        alpha_geom     = deg2rad(alpha_geom_deg(:));  % N x 1

        % Shape vectors for calculation
        alpha0_vec = deg2rad(alpha0_vec_deg(:))';   % 1 x M
        a2d_row    = a2d_vec(:)';                   % 1 x M
        M_elems    = numel(a2d_vec);

        % mu = c * a2d / (4*b)  -> N x M
        mu        = (c_theta(:) * a2d_row) ./ (4 * b);  % N x M
        sin_theta = sin(theta(:));                      % N x 1

        % Build B and RHS (LHS in older code)
        B_all = zeros(N, N, M_elems);
        LHS   = zeros(N,     M_elems);
        for m = 1:M_elems
            B_all(:,:,m) = S_basis .* (1 + (mu(:,m) * K) ./ sin_theta);
            LHS(:,m)     = mu(:,m) .* (alpha_geom - alpha0_vec(m));
        end

        % Solve for A coefficients
        A_all = zeros(N, M_elems);
        for m = 1:M_elems
            A_all(:,m) = B_all(:,:,m) \ LHS(:,m);
        end

        % ---------- CL from spanwise integration (as before) ----------
        sum2     = S_basis * A_all;                 % N x M
        CL_theta = (4 * b) .* (sum2 ./ c_theta(:)); % N x M

        % Build distribution over N+1 sample points (0..b/2)
        y = linspace(0, b/2, N+1);
        CL_dist_per_elem = zeros(M_elems, N+1);
        CL_dist_per_elem(:,1)      = 0;          % at root
        CL_dist_per_elem(:,2:end)  = CL_theta.'; % rows = elements

        % Total across chordwise elements:
        CL_total = sum(CL_dist_per_elem, 1);     % 1 x (N+1)

        % Chord at y sample points (linear taper)
        c_y = Croot + (Ctip - Croot) .* (y / (b/2));

        % Integrate to get overall CL
        CL_wing_total = (2 / S) * trapz(y, CL_total .* c_y);

        % ---------- Induced drag from Fourier coefficients ----------
        % Combine A-coefficients of all chordwise elements
        A_combined = sum(A_all, 2);    % N x 1
        % Prandtl lifting-line relations:
        %   CL ≈ pi * AR * A1
        %   CDi = pi * AR * sum( n * A_n^2 )  with n = harmonic index
        K_col = K(:);                  % N x 1, harmonic indices 1,3,5,...
        CDi = pi * AR * sum( K_col .* (A_combined.^2) );

        % L/D using induced drag only (no extra profile drag model here)
        CL_results(ia, il)     = CL_wing_total;
        CDi_results(ia, il)    = CDi;
        LoverD_results(ia, il) = CL_wing_total / CDi;
        A1_results(ia, il)     = A_combined(1);
    end
end

%% ---------------- BUILD RESULTS TABLE (T) ----------------
rows = [];
for ia = 1:nAR
    for il = 1:nLam
        rows = [rows; ...
            AR_vec(ia), ...
            lambda_vec(il), ...
            i_w_deg, ...
            CL_results(ia, il), ...
            CDi_results(ia, il), ...
            LoverD_results(ia, il)]; %#ok<AGROW>
    end
end

T = array2table(rows, ...
    'VariableNames', {'AR','lambda','i_w_deg','CL_wing','CDi','L_over_D'});

writetable(T, 'parametric_CL_CDi_LD_results.csv');
fprintf('Saved results to parametric_CL_CDi_LD_results.csv (rows = %d)\n', size(T,1));

%% ---------------- CONTOUR MAPS: CL, CD, L/D vs AR & lambda ----------------
[ARg, Lg] = meshgrid(AR_vec, lambda_vec);  % AR along x, lambda along y

CL_plot  = CL_results.';      % rows = lambda, cols = AR
CD_plot  = CDi_results.';     % "
LD_plot  = LoverD_results.';  % "

% ---- 1) CL contour ----
figure('Position',[200 200 780 520]);
contourf(ARg, Lg, CL_plot, 20, 'LineColor','none');
colorbar;
xlabel('Aspect Ratio (AR)');
ylabel('Taper ratio \lambda = c_{tip}/c_{root}');
title(sprintf('C_L contour (S = %.3f m^2, i_w = %g^\\circ, N = %d)', ...
    S, i_w_deg, N));
set(gca, 'YDir', 'normal');
saveas(gcf, 'CL_contour_AR_lambda.png');

% ---- 2) CD contour (induced drag) ----
figure('Position',[220 220 780 520]);
contourf(ARg, Lg, CD_plot, 20, 'LineColor','none');
colorbar;
xlabel('Aspect Ratio (AR)');
ylabel('Taper ratio \lambda = c_{tip}/c_{root}');
title(sprintf('C_D_i contour (S = %.3f m^2, i_w = %g^\\circ, N = %d)', ...
    S, i_w_deg, N));
set(gca, 'YDir', 'normal');
saveas(gcf, 'CDi_contour_AR_lambda.png');

% ---- 3) L/D contour ----
figure('Position',[240 240 780 520]);
contourf(ARg, Lg, LD_plot, 20, 'LineColor','none');
colorbar;
xlabel('Aspect Ratio (AR)');
ylabel('Taper ratio \lambda = c_{tip}/c_{root}');
title(sprintf('L/D contour (S = %.3f m^2, i_w = %g^\\circ, N = %d)', ...
    S, i_w_deg, N));
set(gca, 'YDir', 'normal');
saveas(gcf, 'LoverD_contour_AR_lambda.png');

%% ---------------- ASSUMPTION SUMMARY (OPTIONAL) ----------------
fprintf('\n================ ASSUMPTIONS USED FOR parametric_CL_CDi_LD_results.csv ================\n');
fprintf('Wing area S                = %.3f m^2\n', S);
fprintf('Twist (root -> tip)        = %.1f deg\n', twist_deg);
fprintf('Sweep angle                = %.1f deg (not used in this model)\n', sweep_deg);
fprintf('Fourier terms / control N  = %d\n', N);

fprintf('\nAspect Ratio AR range:   [%.2f : %.2f] step %.2f\n', ...
    AR_vec(1), AR_vec(end), AR_vec(2)-AR_vec(1));
fprintf('Taper ratio lambda range: [%.2f : %.2f] step %.2f\n', ...
    lambda_vec(1), lambda_vec(end), lambda_vec(2)-lambda_vec(1));
fprintf('Wing setting i_w (single): %.1f deg\n', i_w_deg);

fprintf('\nAirfoil 2D data (assumed):\n');
fprintf('  NACA4412:  a2d = %.3f 1/rad,  alpha0 = %.2f deg\n', ...
    a2d_4412, alpha0_4412_deg);
fprintf('  S1223:     a2d = %.3f 1/rad,  alpha0 = %.2f deg\n', ...
    a2d_s1223, alpha0_s1223_deg);
fprintf('======================================================================================\n');
