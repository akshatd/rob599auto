clear all
close all
clc

T = 0.5;
I = eye(3);
zero = zeros(3);
mass = 1;

g = 9.81;

A = [I T*I; zero I];
B = ((1/mass)*[0.5*T^2*I; T*I]);
C = eye(width(A));
D = zeros(height(B),width(B));

Q = [100 0 0 0 0 0;
     0 100 0 0 0 0;
     0 0 5 0 0 0;
     0 0 0 1 0 0;
     0 0 0 0 1 0;
     0 0 0 0 0 100];
R = eye(3)*0.1;

n = size(A, 2);
m = size(B, 2);
p = size(C, 1);
N = 5;

%K_inf = dlqr(A, B, Q, R)
K_2 = -place(A, B, [0.01 0.01 0 0.01 0 0]);

P = dlyap((A+B*K_2)', Q+K_2'*R*K_2);
[F, G] = predict_mats(A, B, N);

[H, L, M] = cost_mats(F, G, Q, R, P);

time_steps = 100;

%Initial States
r0 = [50 70 500]';
v0 = [0 0 -15]';
x0 = [r0; v0];
u0 = zeros(3, 1);

% Rocket state and control input initialization
x = x0;
u = u0;
% minimum distance due to angle of approach
rxy_max = (500/tan(deg2rad(30)))+50;
% State Constraints
xmax = [rxy_max; rxy_max; 500; 20; 20; 15];
xmin = [-rxy_max; -rxy_max; 0; -20; -20; -15];
Px = [eye(n); -eye(n)];
qx = [xmax; -xmin];

% Input constraints
uxy_max = 12*(tan(deg2rad(10)))*4;
umax = [uxy_max ;uxy_max; 12];
umin = [-uxy_max; -uxy_max; 0];
Pu = [eye(m); -eye(m)];
qu = [umax; -umin];

[Pc, qc, Sc] = constraint_mats(F, G, Pu, qu, Px, qx, Px, qx);

k = 1;

while k<=time_steps
    % Disturbances
    w_x = sin(k/50); 
    w_y = cos(k/50);

    Uopt = quadprog(H, L*x, Pc, qc+Sc*x);
    % Check if Uopt is not empty before assigning values
    if isempty(Uopt)
        break
    end
    uopt = Uopt(1:m, :);

    % Update the state with disturbance
    x = A*x + B*(uopt + [w_x w_y 0]');    

    x_mpc(:, k) = x;
    u_mpc(:, k) = uopt;
    k = k + 1;
end

figure;
yyaxis left
plot(x_mpc(1,:), x_mpc(2, :),'b-')
ylabel('x')
yyaxis right
plot(x_mpc(1,:), x_mpc(3, :), 'r-')
grid on
xlabel('z')
ylabel('y')
legend('r_x, r_z', 'r_y, r_z', location='southeast')
title('Position State Space')

figure;
yyaxis left
plot(x_mpc(4,:), x_mpc(5, :),'b-')
ylabel('x-velocity')

yyaxis right
plot(x_mpc(4,:), x_mpc(6, :), 'r-')
grid on
xlabel('z-velocity')
ylabel('y-velocity')
title('Velocity State Space')
legend('v_x, v_z', 'v_y, v_z', location='southeast')

figure;
hold on
grid on
plot(x_mpc(1,:), 'r-')
plot(x_mpc(2,:), 'g-')
plot(x_mpc(3,:), 'b-')

legend('r_x', 'r_y', 'r_z')
title('Position States')
ylabel('Distance [m]')
xlabel('Time Steps T')
hold off

figure;
hold on
grid on
plot(x_mpc(4,:), 'r-')
plot(x_mpc(5,:), 'g-')
plot(x_mpc(6,:), 'b-')

legend('v_x', 'v_y', 'v_z', location='southeast')
title('Velocity States')
ylabel('Velocity [m/s]')
xlabel('Time Steps')
hold off

figure;
hold on
grid on
plot(u_mpc(1,:), 'r-')
plot(u_mpc(2,:), 'g-')
plot(u_mpc(3,:), 'b-')

legend('f_x', 'f_y', 'f_z')
title('Control Input')
ylabel('Thrust [N/kg]')
xlabel('Time Steps')
hold off

figure;
plot3(x_mpc(1,:),x_mpc(2,:),x_mpc(3,:), 'LineWidth',2);
xlabel('r_x');
ylabel('r_y');
zlabel('r_z');
title('Position States');

figure; 
axis([0 600 0 600 0 600])
plot3(x_mpc(1,:),x_mpc(2,:),x_mpc(3,:), 'LineWidth',2);
xlabel('r_x');
ylabel('r_y');
zlabel('r_z');
title('Position States');
hold on 
% Set up the video writer
%video_name = 'Rocket_animation';
%v = VideoWriter([video_name, '.mp4'], 'MPEG-4');
%v.FrameRate = 30;
%open(v);

%for i = 1:length(xs_c(1, :))
%    plot3(xs_c(1, i), xs_c(2, i), xs_c(3, i), '*r', 'LineWidth', 2.5);
%    hold on;

%    frame = getframe(gcf);
%    writeVideo(v, frame);

%    pause(0.01);
%end

% Close the video writer
%close(v);


function [Pc, qc, Sc] = constraint_mats(F,G,Pu,qu,Px,qx,Pxf,qxf)    
    % input dimension
    m = size(Pu,2);
    
    % state dimension
    n = size(F,2);
    
    % horizon length
    N = size(F,1)/n;
    
    % number of input constraints
    ncu = numel(qu);
    
    % number of state constraints
    ncx = numel(qx);
    
    % number of terminal constraints
    ncf = numel(qxf);
    
    % if state constraints exist, but terminal ones do not, then extend the
    % former to the latter
    if ncf == 0 && ncx > 0
        Pxf = Px;
        qxf = qx;
        ncf = ncx;
    end
    
    % Input constraints
    
    % Build "tilde" (stacked) matrices for constraints over horizon
    Pu_tilde = kron(eye(N),Pu);
    qu_tilde = kron(ones(N,1),qu);
    Scu = zeros(ncu*N,n);
    
    % State constraints
    
    % Build "tilde" (stacked) matrices for constraints over horizon
    Px0_tilde = [Px; zeros(ncx*(N-1) + ncf,n)];
    if ncx > 0
        Px_tilde = [kron(eye(N-1),Px) zeros(ncx*(N-1),n)];
    else
        Px_tilde = zeros(ncx,n*N);
    end
    Pxf_tilde = [zeros(ncf,n*(N-1)) Pxf];
    Px_tilde = [zeros(ncx,n*N); Px_tilde; Pxf_tilde];
    qx_tilde = [kron(ones(N,1),qx); qxf];
    
    % Final stack
    if isempty(Px_tilde)
        Pc = Pu_tilde;
        qc = qu_tilde;
        Sc = Scu;
    else
        % eliminate x for final form
        Pc = [Pu_tilde; Px_tilde*G];
        qc = [qu_tilde; qx_tilde];
        Sc = [Scu; -Px0_tilde - Px_tilde*F];
    end
end

function [H,L,M] = cost_mats(F,G,Q,R,P)   
    % Dimensions
    n = size(F,2);
    N = size(F,1)/n;
    
    % Diagonalize Q and R
    Qd = kron(eye(N-1),Q);
    Qd = blkdiag(Qd,P);
    Rd = kron(eye(N),R);
    
    % Hessian
    H = 2*G'*Qd*G + 2*Rd;
    
    % Linear term
    L = 2*G'*Qd*F;
    
    % Constant term
    M = F'*Qd*F + Q;
    
    % Make sure the Hessian is symmetric
    H=(H+H')/2;
end

function [F,G] = predict_mats(A,B,N)   
    % Dimensions
    n = size(A,1);
    m = size(B,2);
    
    F = zeros(n*N,n);
    G = zeros(n*N,m*(N-1));
    
    for i = 1:N
       %  F
       F(n*(i-1)+(1:n),:) = A^i;
       
       % G
       for j = 1:i
           G(n*(i-1)+(1:n),m*(j-1)+(1:m)) = A^(i-j)*B;
       end
    end
end