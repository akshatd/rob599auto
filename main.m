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
plot3(X(:,3), X(:,2), X(:,1));
xlabel('z')
ylabel('y')
zlabel('x')
title('3D Position');


% Get jacobians
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

% check if model is open loop stable
eigAc = eig(Ac);
fprintf('Linearized continuous model is ');
% check if all eigenvalues are in left half plane
notLeftHalfPlane = eigAc(real(eigAc) >= 0);
if size(notLeftHalfPlane) ~= 0
	fprintf('open loop unstable, not all eigenvalues in left half plane:');
	display(notLeftHalfPlane);
else
	fprintf('open loop stable\n');
end

% Plot the eigenvalues in the complex plane
figure(4);
xline(0, 'k', 'DisplayName', 'Half plane Axis');
hold on;
plot(real(eigAc), imag(eigAc), 'x', 'MarkerSize', 10, 'DisplayName', 'Eigenvalues');
xlabel('Real');
ylabel('Imaginary');
axis equal;
title('Eigenvalues of the linearized continuous model');
legend("Location", "best");
xlim([-1.25 1.25]);
ylim([-1.25 1.25]);

% Check controllability
fprintf('Linearized continuous controllability matrix rank: %d vs %d\n', rank(ctrb(Ac, Bc)), size(Ac, 1));

% Discretize the model assunming everything is observable
Ts = 0.01;
sysd = c2d(ss(Ac, Bc, eye(size(Ac)), 0), Ts);
Ad = sysd.A;
Bd = sysd.B;

% Check if the system is open loop stable
eigAd = eig(Ad);
fprintf('Linearized discrete model is ');
% check if all eigenvalues are in unit circle
outsideUnitCircle = eigAd(abs(eigAd) >= 1);
if size(outsideUnitCircle) ~= 0
	fprintf('open loop unstable, eigenvalues outside unit circle:');
	display(outsideUnitCircle);
else
	fprintf('open loop stable\n');
end

% plot the eigenvalues in a unit circle
theta = 0:0.01:2*pi;
figure(5);
plot(cos(theta), sin(theta), 'k', 'DisplayName', 'Unit Circle');
hold on;
plot(real(eigAd), imag(eigAd), 'x', 'MarkerSize', 10, 'DisplayName', 'Eigenvalues');
xlabel('Real');
ylabel('Imaginary');
axis equal;
title('Eigenvalues of the linearized discrete model');
legend("Location", "best");
xlim([-1.25 1.25]);
ylim([-1.25 1.25]);

% Check Controllablitity of the discrete model
fprintf('Linearized discrete controllability matrix rank: %d vs %d\n', rank(ctrb(Ad, Bd)), size(Ad, 1));