clc; clear; close all;

m0 = 100000;

% Initial state: X = [x y z u v w phi theta psi p q r m]
X0 = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; m0];
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


% linearize model
% syms x y z u v w phi theta psi p q r m
% X_sym = [x; y; z; u; v; w; phi; theta; psi; p; q; r; m];
% syms
