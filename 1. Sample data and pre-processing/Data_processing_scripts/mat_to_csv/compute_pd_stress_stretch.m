function R = compute_pd_stress_stretch(S, analysis, pd_tag)
% Compute stress–strain ONLY for a single Pd test (pd_tag in {-5,0,5}).
% Thin-wall stress formulas for quick visualization. Auto-finds the measurement block.

mmHg_to_kPa = 0.13332;
grams_to_mN = 9.80665;

[blk, ~] = find_measurement_block(S);
if isempty(blk)
    error('Could not locate a block with fields L,H,OD and biaxial_* inside the .mat file.');
end
num = map_pd_tag_to_index(pd_tag);
if num < 1
    error('Unsupported Pd tag: %g (expected -5, 0, or 5)', pd_tag);
end
pdfield = sprintf('test%d', num);
assert(isfield(blk.biaxial_Pd, pdfield), 'Missing Pd field: %s', pdfield);
data = blk.biaxial_Pd.(pdfield);   % columns: [P(mmHg), od(mm), Ft(g), lambda_z, l(mm)]

% ---- Geometry (unloaded)
L  = blk.L;   % mm
H  = blk.H;   % mm
OD = blk.OD;  % mm
ID = OD - 2*H;
OR = OD/2; 
IR = ID/2;

% ---- Reference volume (mm^3)
ref_vol = pi/4 * (OD^2 - ID^2) * L;

% ---- Grab the specific Pd table

% ---- Clean NaNs in od
data = data(isfinite(data(:,2)),:);

% ---- Loaded quantities
od      = data(:,2);        % mm
lambdaz = data(:,4);        % -
l       = abs(data(:,5));   % mm

% inner diameter from volume constraint (clip small negatives)
id_sq = od.^2 - (4*ref_vol)./(pi*l);
index_neg = find(id_sq<1);

% ---- Clean neg id 
data(index_neg,:) =[]

% ---- Loaded quantities
od      = data(:,2);        % mm
lambdaz = data(:,4);        % -
l       = abs(data(:,5));   % mm

id_sq = od.^2 - (4*ref_vol)./(pi*l);
id      = sqrt(id_sq);          % mm
or      = od/2; 
ir      = id/2;                 % mm
h       = or - ir;              % mm

% ---- Experimental series
Pi_exp = mmHg_to_kPa * data(:,1);     % kPa
Ft_exp = grams_to_mN * data(:,3);     % mN
Fv_exp = Ft_exp + pi*ir.^2.*Pi_exp;   % mN

% ---- Thin-wall stresses (2D)
if strcmpi(analysis,'2D')
    tzz_exp = Fv_exp ./ (pi*(or.^2 - ir.^2)); % kPa
    tqq_exp = (Pi_exp .* ir) ./ h;            % kPa
   %lambdaq = (or + ir) ./ (OR + IR);         % mid-radius stretch proxy
   lambdaq = (or) ./ (OR);         % mid-radius stretch proxy
else
    tzz_exp = [];
    tqq_exp = [];
    lambdaq = [];
end

% ---- Package
R = struct();
R.pd_tag    = pd_tag;
R.test_idx  = num;

% geometry (unloaded)
R.L         = L;
R.H         = H;
R.OD        = OD;
R.ID        = ID;
R.OR        = OR;
R.IR        = IR;
R.ref_vol   = ref_vol;

% loaded kinematics
R.lambdaz   = lambdaz;
R.lambdaq   = lambdaq;    % [] if analysis ~= '2D'
R.l         = l;          % mm
R.od        = od;         % mm
R.or        = or;         % mm
R.ir        = ir;         % mm
R.h         = h;          % mm

% forces/pressures
R.Pi_exp    = Pi_exp;     % kPa
R.Ft_exp    = Ft_exp;     % mN
R.Fv_exp    = Fv_exp;     % mN

% stresses (2D)
R.tzz_exp   = tzz_exp;    % kPa
R.tqq_exp   = tqq_exp;    % kPa
end
% ===== Helpers ============================================================

function [blk, blkname] = find_measurement_block(S)
% Recursively search struct S for a nested block that has
%   L, H, OD, biaxial_Pd, biaxial_Fl
% Returns that block and its dotted path name. Empty if not found.

    want = {@(x)isfield(x,'L'), @(x)isfield(x,'H'), @(x)isfield(x,'OD'), ...
            @(x)isfield(x,'biaxial_Pd'), @(x)isfield(x,'biaxial_Fl')};

    [blk, blkname, found] = search_any(S, '', want);
    if ~found
        blk = [];
        blkname = '';
    end
end

function [hit, path, found] = search_any(X, prefix, want)
% Depth-first search through nested structs.
% Returns first struct that satisfies all predicates in cell array 'want'.

    hit   = [];
    path  = '';
    found = false;

    if ~isstruct(X)
        return;
    end

    % If X itself satisfies all requirements, return it
    ok = true;
    for i = 1:numel(want)
        if ~want{i}(X)
            ok = false; break;
        end
    end
    if ok
        hit   = X;
        path  = prefix;
        found = true;
        return;
    end

    % Otherwise, recurse into fields
    f = fieldnames(X);
    for i = 1:numel(f)
        child = X.(f{i});
        childPrefix = f{i};
        if ~isempty(prefix)
            childPrefix = sprintf('%s.%s', prefix, f{i});
        end
        [hit, path, found] = search_any(child, childPrefix, want);
        if found, return; end
    end
end

function num = map_pd_tag_to_index(tag)
% Map Pd tags {-5, 0, 5} -> test indices {1,2,3}
    switch tag
        case -5, num = 1;
        case  0, num = 2;
        case  5, num = 3;
        otherwise, num = -1; % caller should error on <1
    end
end

function num = map_fl_tag_to_index(tag)
% Map Fl tags {10, 60, 100, 140} -> test indices {1,2,3,4}
    switch tag
        case 10,  num = 1;
        case 60,  num = 2;
        case 100, num = 3;
        case 140, num = 4;
        otherwise, num = -1; % caller should error on <1
    end
end
