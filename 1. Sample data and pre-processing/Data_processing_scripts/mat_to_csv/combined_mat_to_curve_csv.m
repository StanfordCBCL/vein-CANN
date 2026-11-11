% BATCH script: save one CSV per Pd test (-5,0,5) and per Fl test (10,60,100,140)
% Pd files: <Specimen>_pd95.csv, <Specimen>_pd100.csv, <Specimen>_pd105.csv
% Fl files: <Specimen>_fl10.csv, <Specimen>_fl60.csv, <Specimen>_fl100.csv, <Specimen>_fl140.csv

clear; clc;

specimens = {
     'Rosie',  'Rosie.mat';
    % 'Pablo',  'Pablo.mat';
    %'Audi',  'Audi.mat';
   % 'Dodge', 'Dodge.mat';
   % 'Gus', 'Gus.mat';
    %'Lexus', 'Lexus.mat';
     'Lucid', 'Lucid.mat';
   % 'Mojito', 'Mojito.mat';
    %'Mercedes', 'Mercedes.mat';
    % 'Orion', 'Orion.mat';
     %'Martini', 'Martini.mat';
    % 'Sagitarius', 'Sagitarius.mat'

};

analysis  = '2D';                 % keep as '2D' for σθ/σz vs stretches
tests_Pd  = [-5, 0, 5];           % will be labeled Pd95, Pd100, Pd105 (100+tag)
tests_Fl  = [10, 60, 100, 140];

outdir = fullfile(pwd, 'out');
if ~exist(outdir, 'dir'), mkdir(outdir); end

% Color per specimen; marker per test
colors  = lines(size(specimens,1));                 % one color per specimen
pd_mark = {'o','s','^'};                            % markers for Pd95/100/105
fl_mark = {'o','s','^','d'};                        % markers for Fl10/60/100/140

% Keep handles for legends
pd_handles = []; pd_labels = {};
fl_handles = []; fl_labels = {};

% ---------- PER-SPECIMEN EXPORTS + COLLECT PLOT DATA ----------
for k = 1:size(specimens,1)
    name   = specimens{k,1};
    matpth = specimens{k,2};

    fprintf('==> %s\n', name);
    S = load(matpth);
    if exist('S','var') && isstruct(S) && isfield(S,'S')
        S = S.S;   % unwrap one level
    end

    % ----- Pd tests -----
    for t = 1:numel(tests_Pd)
        tag = tests_Pd(t);

        % compute & write CSV
        R = compute_pd_stress_stretch(S, analysis, tag);
        baseP  = 100 + tag;                 % 95,100,105
        csv_pd = fullfile(outdir, sprintf('%s_pd%d.csv', name, baseP));

        if strcmpi(analysis,'2D')
            T = table(R.lambdaq(:), R.tqq_exp(:), R.lambdaz(:), R.tzz_exp(:), ...
                      R.Pi_exp(:), R.od(:), R.l(:), R.Ft_exp(:), R.Fv_exp(:), ...
                      'VariableNames', {'lambda_theta','sigma_theta_kPa','lambda_z','sigma_z_kPa', ...
                                        'pressure_kPa','od_mm','length_mm','F_transducer_mN','F_vessel_mN'});
        else
            T = table(R.Pi_exp(:), R.lambdaz(:), R.od(:), R.l(:), R.Ft_exp(:), R.Fv_exp(:), ...
                      'VariableNames', {'pressure_kPa','lambda_z','od_mm','length_mm','F_transducer_mN','F_vessel_mN'});
        end
        writetable(T, csv_pd);
        fprintf('   wrote %s\n', csv_pd);

        % Store/plot Pd (σθ vs λθ)
        if strcmpi(analysis,'2D') && ~isempty(R.lambdaq)
            if t > numel(pd_mark), mk = 'o'; else, mk = pd_mark{t}; end
            h = plot_pd_lazy(R.lambdaq, R.tqq_exp, colors(k,:), mk);
            pd_handles(end+1) = h; 
            pd_labels{end+1}  = sprintf('%s Pd%d', name, baseP); 
        end
    end

    % ----- Fl tests -----
    for t = 1:numel(tests_Fl)
        tag = tests_Fl(t);

        % compute & write CSV
        R = compute_fl_stress_stretch(S, analysis, tag);
        csv_fl = fullfile(outdir, sprintf('%s_fl%d.csv', name, tag));

        if strcmpi(analysis,'2D')
            T = table(R.lambdaq(:), R.tqq_exp(:), R.lambdaz(:), R.tzz_exp(:), ...
                      R.Pi_exp(:), R.od(:), R.l(:), R.Ft_exp(:), R.Fv_exp(:), ...
                      'VariableNames', {'lambda_theta','sigma_theta_kPa','lambda_z','sigma_z_kPa', ...
                                        'pressure_kPa','od_mm','length_mm','F_transducer_mN','F_vessel_mN'});
        else
            T = table(R.Pi_exp(:), R.lambdaz(:), R.od(:), R.l(:), R.Ft_exp(:), R.Fv_exp(:), ...
                      'VariableNames', {'pressure_kPa','lambda_z','od_mm','length_mm','F_transducer_mN','F_vessel_mN'});
        end
        writetable(T, csv_fl);
        fprintf('   wrote %s\n', csv_fl);

        % Store/plot Fl (σz vs λz)
        if strcmpi(analysis,'2D') && ~isempty(R.lambdaz)
            if t > numel(fl_mark), mk = 'o'; else, mk = fl_mark{t}; end
            h = plot_fl_lazy(R.lambdaz, R.tzz_exp, colors(k,:), mk);
            fl_handles(end+1) = h; 
            fl_labels{end+1}  = sprintf('%s Fl%d', name, tag); 
        end
    end
end

% ---------- FINALIZE & SAVE COMBINED PLOTS ----------
% Pd combined
if ~isempty(pd_handles)
    fPd = ancestor(pd_handles(1), 'figure');
    figure(fPd); box on;
    xlabel('\lambda_\theta'); ylabel('\sigma_\theta (kPa)');
    title('Pd Tests – Circumferential Stress–Stretch');
    legend(pd_handles, pd_labels, 'Location','bestoutside');
    saveas(fPd, fullfile(outdir, 'combined_Pd_plot.png'));
end

% Fl combined
if ~isempty(fl_handles)
    fFl = ancestor(fl_handles(1), 'figure');
    figure(fFl); box on;
    xlabel('\lambda_z'); ylabel('\sigma_z (kPa)');
    title('Fl Tests – Axial Stress–Stretch');
    legend(fl_handles, fl_labels, 'Location','bestoutside');
    saveas(fFl, fullfile(outdir, 'combined_Fl_plot.png'));
end

fprintf('Saved plots to: %s\n', outdir);

% ---------- local plotting helpers (avoid clutter above) ----------
function h = plot_pd_lazy(x, y, c, mk)
    persistent fPd
    if isempty(fPd) || ~ishandle(fPd)
        fPd = figure('Color','w'); hold on;
    else
        figure(fPd); hold on;
    end
    h = plot(x, y, 'LineStyle','-', 'Color', c, 'Marker', mk, 'DisplayName','');
end

function h = plot_fl_lazy(x, y, c, mk)
    persistent fFl
    if isempty(fFl) || ~ishandle(fFl)
        fFl = figure('Color','w'); hold on;
    else
        figure(fFl); hold on;
    end
    h = plot((x), (y), 'LineStyle','-', 'Color', c, 'Marker', mk, 'DisplayName','');
end
