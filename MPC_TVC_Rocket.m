clc; clear; close all;

m0 = 10000;
X0 = [0; 0; 0; 10; 0; 0; 0; 0; 0; 0; 0; 0; m0];

U0 = [0; 0; 1e5];

% waypoints
waypoints = [100, 0, 0;
    200, 0, 0;
    300, 0, 0;
    400, 0, 0;
    500, 0, 0];
num_waypoints = size(waypoints, 1);
waypoint_index = 1;
tolerance = 5;

% Simulation parameters
t_end = 10;       % seconds for each segment
ts = 0.01;        % discrete time step
tspan = 0:ts:t_end;

% Store simulation data
Xk_hist = zeros(length(X0), length(tspan));
Xk_hist(:,1) = X0;

% Symbolic states & inputs for Jacobians
syms x y z u v w phi theta psi p q r m mu_pitch mu_yaw T real
X_sym = [x; y; z; u; v; w; phi; theta; psi; p; q; r; m];
U_sym = [mu_pitch; mu_yaw; T];

% 6-DOF rocket EOM
f_sym = sixDOF_EOM_STVCR(0, X_sym, U_sym);

% Jacobians
A_sym = jacobian(f_sym, X_sym);
B_sym = jacobian(f_sym, U_sym);

% Evaluate numerically at (X0, U0)
Ac = double(subs(A_sym, [X_sym; U_sym], [X0; U0]));
Bc = double(subs(B_sym, [X_sym; U_sym], [X0; U0]));

% Discretize
sysd = c2d(ss(Ac, Bc, eye(size(Ac)), 0), ts);
Ad = sysd.A;
Bd = sysd.B;

% MPC Params
N = 10;
nx = size(Ad,1);
nu = size(Bd,2);

% Q and R
Q_dia = [100, 100, 100,  0, 0, 0,  0, 0, 0,  0, 0, 0,  0];
Q = diag(Q_dia);
R_dia = [0.01, 0.01, 0.001];
R = diag(R_dia);

Xk = X0;
Uk = U0;
u_prev = U0;

for seg = 1:num_waypoints
    
    x_ref = waypoints(seg,1);
    y_ref = waypoints(seg,2);
    z_ref = waypoints(seg,3);
    
    
    for i = 2:length(tspan)
        % Evaluate numerically at (X0, U0)
        Ac = double(subs(A_sym, [X_sym; U_sym], [Xk; Uk]));
        Bc = double(subs(B_sym, [X_sym; U_sym], [Xk; Uk]));
        
        % Discretize
        sysd = c2d(ss(Ac, Bc, eye(size(Ac)), 0), ts);
        Ad = sysd.A;
        Bd = sysd.B;
        
        
        Uk = mpc_control(Xk, u_prev, x_ref, y_ref, z_ref, Ad, Bd, Q, R, N );
        
        [~, X_temp] = ode45(@(t, X) sixDOF_EOM_STVCR(t, X, Uk), [0 ts], Xk);
        Xk = X_temp(end, :)';
        Xk_hist(:,i) = Xk;
        
        u_prev = Uk;
        
        % Check if rocket is near the current waypoint
        dist_to_wp = norm([Xk(1) - x_ref, Xk(2) - y_ref, Xk(3) - z_ref]);
        if dist_to_wp < tolerance
            fprintf('   Reached waypoint #%d in %.2f s\n', seg, tspan(i));
            break; % proceed to next waypoint
        end
        
    end
end

% Plot Results
figure; hold on;
plot3(waypoints(:,2), waypoints(:,3), waypoints(:,1), 'ro-', 'LineWidth', 2);
plot3(Xk_hist(2,:), Xk_hist(3,:), Xk_hist(1,:), 'b', 'LineWidth', 2);
xlabel('X'); ylabel('Y'); zlabel('Z');
legend('Target Waypoints','Rocket Trajectory');
title('Rocket MPC Tracking Multiple Waypoints (Extended-State MPC)');
grid on; view(3);


function Uk = mpc_control(Xk, u_prev, x_ref, y_ref, z_ref, Ad, Bd, Q_e, R, N)
nx = size(Ad,1); %13
nu = size(Bd,2); %3

r_vec = [x_ref; y_ref; z_ref; zeros(nx-3,1)];

n_ext = nx + nu + nx;

A_ext = [ Ad,           Bd,            zeros(nx, nx);
    zeros(nu, nx), eye(nu),       zeros(nu, nx);
    zeros(nx, nx), zeros(nx, nu), eye(nx) ];
B_ext = [ Bd; eye(nu); zeros(nx, nu) ];

E = [ eye(nx), zeros(nx,nu), -eye(nx) ];
Q = E' * Q_e * E;

x_ext0 = [ Xk; u_prev; r_vec];

% Build prediction matrices F and G for the horizon N:
F = zeros(n_ext*N, n_ext);
G = zeros(n_ext*N, nu*N);
for i = 1:N
    F((i-1)*n_ext+1:i*n_ext, :) = A_ext^i;
    for j = 1:i
        G((i-1)*n_ext+1:i*n_ext, (j-1)*nu+1:j*nu) = A_ext^(i-j)*B_ext;
    end
end

% Build block-diagonal weighting matrices:
Q_bar = kron(eye(N), Q);   % weight on extended states
R_bar = kron(eye(N), R);   % weight on control increments

% Condensed cost function:
% J = (F*x_ext0 + G*DeltaU)'*Q_bar*(F*x_ext0 + G*DeltaU) + DeltaU'*R_bar*DeltaU.
% When expanded and arranged in quadprog form (0.5*DeltaU'*H*DeltaU + f'*DeltaU),
% we have:
H = 2*(G' * Q_bar * G + R_bar);
f = 2*(F * x_ext0)' * Q_bar * G;  % f is a row vector; quadprog needs a column vector (we take its transpose later).

% Input constraints: u = u_prev + DeltaU_k must lie between u_min and u_max.
u_min  = [-pi/6; -pi/6; 0];       % minimum allowed control
u_max  = [ pi/6;  pi/6; 1e8];      % maximum allowed control

% Build inequality matrices for the inputs:
% For each time step, we require:
%   u_min - u_prev <= DeltaU_k <= u_max - u_prev.
Aineq_input = [ eye(nu*N); -eye(nu*N) ];
Bineq_input = [ repmat(u_max - u_prev, N, 1);
    -repmat(u_min - u_prev, N, 1) ];

% No state constraints; so we use only the input constraints:
Aineq = Aineq_input;
Bineq = Bineq_input;

% Solve QP using quadprog.
% Note: quadprog expects the cost function f as a column vector.
options = optimoptions('quadprog','Display','off');
DeltaU_opt = quadprog(H, f', [], [], [], [], [], [], [], options);

% Extract the first control move (DeltaU_opt is stacked over N steps)
dU0 = DeltaU_opt(1:nu);
Uk = u_prev + dU0;
end


function dXdt = sixDOF_EOM_STVCR(~, X, U)

% Extract states
x    = X(1);   y     = X(2);   z    = X(3);
u    = X(4);   v     = X(5);   w    = X(6);
phi  = X(7);   theta = X(8);   psi  = X(9);
p    = X(10);  q     = X(11);  r    = X(12);
m    = X(13);

% Control inputs
mu1 = U(1);
mu2 = U(2);
T   = U(3);

% Constants
g = 9.81;
rho = 1.225; % air density
D = 3;       % diameter
L = 25;      % length
Sa = pi*(0.5*D)^2;  % frontal area
Sy = L*D;
Sn = Sy;            % same as Sy
Isp = 300;          % specific impulse

% Moments of inertia
Jx = 0.5*m*(0.5*D)^2;
Jy = 0.25*m*(0.5*D)^2 + (1/12)*m*L^2;
Jz = Jy;

% Aero angles
alpha = atan2(w, u);
beta  = atan2(v, sqrt(u^2 + w^2));

% Aero coeffs (heuristic)
CA = 0.8;
CY = 0.1*alpha;
CN = 0.1*beta;
Cl = 0; Cm = 0; Cn = 0;

% Dynamic pressure
v_rel = sqrt(u^2 + v^2 + w^2);
dyP = 0.5 * rho * v_rel^2;

sphi = sin(phi);     cphi = cos(phi);
stheta = sin(theta); ctheta = cos(theta);
spsi = sin(psi);     cpsi = cos(psi);
smu1 = sin(mu1);     cmu1 = cos(mu1);
smu2 = sin(mu2);     cmu2 = cos(mu2);
ttheta = tan(theta);


%dx = u;
%dy = v;
%dz = w;

dx = (u * (ctheta*cpsi)) + (v * (sphi*stheta*cpsi - cphi*spsi)) + (w * (cphi*stheta*cpsi + sphi*spsi));
dy = (u * (ctheta*spsi)) + (v * (sphi*stheta*spsi + cphi*cpsi)) + (w * (cphi*stheta*spsi - sphi*cpsi));
dz = (u * (-stheta)) + (v * (sphi*ctheta)) + (w * (cphi*ctheta));

du = -g*(ctheta*cpsi) - (dyP/m)*Sa*CA + (T/m)*cmu1*cmu2 - q*w + r*v;
dv = -g*(sphi*stheta*cpsi - cphi*spsi) + (dyP/m)*Sy*CY ...
    - (T/m)*cmu1*smu2 - r*u + p*w;
dw = -g*(cphi*stheta*cpsi + sphi*spsi) - (dyP/m)*Sn*CN ...
    - (T/m)*smu1 - p*v + q*u;

dphi   = p + q*sphi*ttheta + r*cphi*ttheta;
dtheta = q*cphi - r*sphi;
dpsi   = (q*sphi + r*cphi)/ctheta;

dp = (dyP*Sa*D*Cl)/Jx;
dq = (dyP*Sy*D*Cm - T*smu1*L)/Jy;
dr = (dyP*Sn*D*Cn - T*cmu1*smu2*L)/Jz;

dm = - T/(Isp*g);

dXdt = [dx; dy; dz; du; dv; dw; dphi; dtheta; dpsi; dp; dq; dr; dm];
end
