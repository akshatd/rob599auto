clc; clear; close all;

m0 = 100000;

% Initial state: X = [x y z u v w phi theta psi p q r m]
X0 = [0; 0; 0; 0.1; 0; 0; 0; 0; 0; 0; 0; 0; m0];
U0 = [0; 0];
% Time span
tspan = [0 50];

% ODE45 Solver
[t, X] = ode15s(@(t, X) sixDOF_EOM_STVCR(t, X, U0), tspan, X0);

% Plot results
figure(1);
% Velocity plots
subplot(3,1,1);
plot(t, X(:,4:6)); legend('u', 'v', 'w'); title('Velocities');
% Angular rate plots
subplot(3,1,2);
plot(t, X(:,7:9)); legend('p', 'q', 'r'); title('Angular Rates');
% Euler angles plots
subplot(3,1,3);
plot(t, X(:,10:12)); legend('\phi', '\theta', '\psi'); title('Euler Angles');

figure(2);
% Position plots
subplot(2,1,1);
plot(X(:,2), X(:,1));
ylabel('x')
xlabel('y')
title('Position');

subplot(2,1,2);
plot(X(:,3), X(:,1));
ylabel('x')
xlabel('z')
title('Position');

% 3D position plot
figure(3);
plot3(X(:,1), X(:,2), X(:,3));
xlabel('x')
ylabel('y')
zlabel('z')
title('3D Position');


% Linearize model
syms x y z u v w phi theta psi p q r m
X_sym = [x; y; z; u; v; w; phi; theta; psi; p; q; r; m];
syms mu_pitch mu_yaw
U_sym = [mu_pitch; mu_yaw];

f = sixDOF_EOM_STVCR(0, X_sym, U_sym);
A = jacobian(f, X_sym);
B = jacobian(f, U_sym);

% Linearize model at the initial condition of zeros
Ac = double(subs(A, [X_sym; U_sym], [X0; U0]));
Bc = double(subs(B, [X_sym; U_sym], [X0; U0]));

% Check controllability
fprintf('Linearized continuous controllability matrix rank: %d\n vs %d\n', rank(ctrb(Ac, Bc)), size(Ac, 1));

% Discretize the model assunming everything is observable
Ts = 0.1;
sysd = c2d(ss(Ac, Bc, eye(size(Ac)), 0), Ts);
Ad = sysd.A;
Bd = sysd.B;

% Check if the system is open loop stable
eigAd = eig(Ad);
fprintf('Discrete time model is ');
% check if all eigenvalues are in unit circle
outsideUnitCircle = eigAd(abs(eigAd) >= 1);
if size(outsideUnitCircle) ~= 0
	fprintf('open loop unstable, eigenvalues outside unit circle:');
	display(outsideUnitCircle);
else
	fprintf('open loop stable\n');
end

% Check Controllablitity of the discrete model
fprintf('Linearized discrete controllability matrix rank: %d\n vs %d\n', rank(ctrb(Ad, Bd)), size(Ad, 1));