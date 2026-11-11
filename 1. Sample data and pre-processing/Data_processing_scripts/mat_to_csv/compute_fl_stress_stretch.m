function R = compute_fl_stress_stretch(S, analysis, fl_tag)
% Compute stress–strain ONLY for a single F–L test (fl_tag in {10,60,100,140}).
% Thin-wall stress formulas (same as Pd version). Auto-finds the measurement block.

mmHg_to_kPa = 0.13332;
grams_to_mN = 9.80665;

[blk, ~] = find_measurement_block(S);
if isempty(blk)
    error('Could not locate a block with fields L,H,OD and biaxial_* inside the .mat file.');
end

% Geometry
L  = blk.L;   % mm
H  = blk.H;   % mm
OD = blk.OD;  % mm
ID = OD - 2*H;
OR = OD/2; IR = ID/2;

% Reference volume (mm^3)
ref_vol = pi/4 * (OD^2 - ID^2) * L;

% Grab the specific Fl table
num = map_fl_tag_to_index(fl_tag);
if num < 1
    error('Unsupported Fl tag: %g (expected 10, 60, 100, or 140)', fl_tag);
end
flfield = sprintf('test%d', num);
assert(isfield(blk.biaxial_Fl, flfield), 'Missing Fl field: %s', flfield);
data = blk.biaxial_Fl.(flfield);   % columns: [P(mmHg), od(mm), Ft(g), lambda_z, l(mm)]

% Clean NaNs in od
data = data(isfinite(data(:,2)),:);

% Loaded quantities
od      = data(:,2);        % mm
lambdaz = data(:,4);        % -
l       = (data(:,5));   % mm
id      = sqrt(od.^2 - (4*ref_vol)./(pi*l));  % mm
or      = od/2; ir = id/2;  % mm
h       = or - ir;          % mm

% Experimental series
Pi_exp = mmHg_to_kPa * data(:,1);     % kPa
Ft_exp = grams_to_mN * data(:,3);     % mN
Fv_exp = Ft_exp + pi*ir.^2.*Pi_exp;   % mN

% Thin-wall stresses (2D)
if strcmpi(analysis,'2D')
    tzz_exp = Fv_exp ./ (pi*(or.^2 - ir.^2)); % kPa
    tqq_exp = (Pi_exp .* ir) ./ h;            % kPa
    lambdaq = (or ) ./ (OR );         % mid-radius stretch proxy
else
    tzz_exp = []; tqq_exp = []; lambdaq = [];
end

% Package
R = struct();
R.Pi_exp   = Pi_exp;
R.Ft_exp   = Ft_exp;
R.Fv_exp   = Fv_exp;
R.lambdaz  = lambdaz;
R.lambdaq  = lambdaq;
R.od       = od;
R.l        = l;
R.tzz_exp  = tzz_exp;
R.tqq_exp  = tqq_exp;
end

function [blk, blkname] = find_measurement_block(S)
% Find the first nested struct with fields L,H,OD and substructs biaxial_Pd, biaxial_Fl
blk = []; blkname = '';
f1 = fieldnames(S);
for i = 1:numel(f1)
    A = S.(f1{i});
    if ~isstruct(A), continue; end
    f2 = fieldnames(A);
    for j = 1:numel(f2)
        B = A.(f2{j});
        if ~isstruct(B), continue; end
        needed = isfield(B, 'L') && isfield(B,'H') && isfield(B,'OD') && ...
                 isfield(B,'biaxial_Pd') && isfield(B,'biaxial_Fl');
        if needed
            blk = B; blkname = sprintf('%s.%s', f1{i}, f2{j});
            return;
        end
    end
end
end

function num = map_pd_tag_to_index(tag)
% Map Pd tags {-5,0,5} -> test{1,2,3}
switch tag
    case -5, num = 1;
    case  0, num = 2;
    case  5, num = 3;
    otherwise, num = -1;
end
end

function num = map_fl_tag_to_index(tag)
% Map Fl tags {10,60,100,140} -> test{1..4}
switch tag
    case 10,  num = 1;
    case 60,  num = 2;
    case 100, num = 3;
    case 140, num = 4;
    otherwise, num = -1;
end
end
