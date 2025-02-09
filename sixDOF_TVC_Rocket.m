clc; clear; close all;

m0 = 100000;

% Initial state: X = [x y z u v w phi theta psi p q r m]
X0 = [0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; m0];

% Time span
tspan = [0 50];

% ODE45 Solver
[t, X] = ode15s(@(t, X) sixDOF_EOM_STVCR(t, X), tspan, X0);

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


% 6 Degree of Freedom Equations of Motion for Simple TVC Rocket
% **Implemented using EULER ANGLES**

% X = [ x y z u v w phi theta psi p q r m]

function dXdt = sixDOF_EOM_STVCR(t, X)

    % Control Input --- Constant for Open Loop Simulation
    mu1 = 0;
    mu2 = 0;
    T = 1000000;

    % Extract State Variables
    x = X(1);
    y = X(2);
    z = X(3);  
    u = X(4);
    v = X(5);
    w = X(6);
    phi = X(7);
    theta = X(8);
    psi = X(9);
    p = X(10);
    q = X(11);
    r = X(12);
    m = X(13);
    
    % Model Parameters
    g = 9.81; % m/s

    % Rocket Parameters
    D = 3; % m ----- TODO: Make this according to the rocket model
    L = 25; % m ----- TODO: Make this according to the rocket model
    Sa = pi * (0.5*D)^2; % m^2
    Sy = L*D; % m^2
    Sn = Sy; % m^2
    % Moments of Inertia
    Jx = 0.5 * m * (0.5 * D)^2;
    Jy = (0.25 * m * (0.5 * D)^2) + ((1/12) * m * L^2);
    Jz = Jy;
    % Specific Impulse
    Isp = 300;
    
    % Aerodynamic Coefficients
    alpha = atan2(w, u);
    beta = atan2(v, sqrt(u^2 + w^2));
    CA = 0.8; % ----- HEURISTIC VALUES | NOT ACCURATE
    CY = 0.1*alpha; % ----- HEURISTIC VALUES | NOT ACCURATE
    CN = 0.1*beta; % ----- HEURISTIC VALUES | NOT ACCURATE
    Cl = 0; % ----- HEURISTIC VALUES | NOT ACCURATE
    Cm = 0; % ----- HEURISTIC VALUES | NOT ACCURATE
    Cn = 0; % ----- HEURISTIC VALUES | NOT ACCURATE
    

    % Dynamic Pressure
    rho = 1.225; % kg/m^3 ----- TODO: switch to altitude dependant model
    v_rel = sqrt(u^2 + v^2 + w^2);
    dyP = 0.5 * rho * v_rel^2;

    % Euler Angles
    sphi = sin(phi);
    cphi = cos(phi);
    stheta = sin(theta);
    ctheta = cos(theta);
    ttheta = tan(theta);
    spsi = sin(psi);
    cpsi = cos(psi);

    % Thrust Angles
    smu1 = sin(mu1);
    cmu1 = cos(mu1);
    smu2 = sin(mu2);
    cmu2 = cos(mu2);
    
    % Kinematics
    dx = (u * (ctheta*cpsi)) + (v * (sphi*stheta*cpsi - cphi*spsi)) + (w * (cphi*stheta*cpsi + sphi*spsi));
    dy = (u * (ctheta*spsi)) + (v * (sphi*stheta*spsi + cphi*cpsi)) + (w * (cphi*stheta*spsi - sphi*cpsi));
    dz = (u * (-stheta)) + (v * (sphi*ctheta)) + (w * (cphi*ctheta));

    % Dynamics
    du = (-g*(ctheta*cpsi)) - ((dyP/m)*Sa*CA) + ((T/m)*cmu1*cmu2) - (q*w) + (r*v);
    dv = (-g*(sphi*stheta*cpsi - cphi*spsi)) + ((dyP/m)*Sy*CY) - ((T/m)*cmu1*smu2) - (r*u) + (p*w);
    dw = (-g*(cphi*stheta*cpsi + sphi*spsi)) - ((dyP/m)*Sn*CN) - ((T/m)*smu1) - (p*v) + (q*u);
    dphi = p + q*sphi*ttheta + r*cphi*ttheta;
    dtheta = q*cphi - r*sphi;
    dpsi = (q*sphi + r*cphi) / ctheta;
    dp = (dyP*Sa*D*Cl) / Jx;
    dq = (dyP*Sy*D*Cm - T*smu1*L) / Jy;
    dr = (dyP*Sn*D*Cn - T*cmu1*smu2*L) / Jz;

    % Mass Change from Tsiolkovsky Rocket Eqn.
    dm = - T / (Isp * g);
    
    % Output the state derivative
    dXdt = [dx; dy; dz; du; dv; dw; dphi; dtheta; dpsi; dp; dq; dr; dm];
end