% 画图 Demo
% Author: Wenhan zhang
% Date: 2022, Oct 10

clc;
clear;
close all;

%%
% reward
reward = readmatrix ('../results/u_max_005_2/TD3_SunPointFaultSatellite_5.csv');
reward = reward(2:end, 2);

% eval
Time = 200;
Ts = 0.1;
tspan = Ts:Ts:Time;

% fault free
eval_res_fault_free = readmatrix ('../results/u_max_005/eval_res_fault_free.csv');
omega_e_fault_free = eval_res_fault_free(2:end, 1:3) * 180 / pi;
s_e_fault_free = eval_res_fault_free(2:end, 4:5);
s_e_fault_free = [s_e_fault_free zeros(size(s_e_fault_free, 1), 1)];
dyn_est_err_fault_free = eval_res_fault_free(2:end, 6:8);
u_fault_free = eval_res_fault_free(2:end, 9:11);
angle_fault_free = eval_res_fault_free(2:end, 12);

% f1
eval_res_f1 = readmatrix ('../results/u_max_005/eval_res_f1.csv');
omega_e_f1 = eval_res_f1(2:end, 1:3) * 180 / pi;
s_e_f1 = eval_res_f1(2:end, 4:5);
s_e_f1 = [s_e_f1 zeros(size(s_e_f1, 1), 1)];
dyn_est_err_f1 = eval_res_f1(2:end, 6:8);
u_f1 = eval_res_f1(2:end, 9:11);
angle_f1 = eval_res_f1(2:end, 12);

% f2
eval_res_f2 = readmatrix ('../results/u_max_005/eval_res_f2.csv');
omega_e_f2 = eval_res_f2(2:end, 1:3) * 180 / pi;
s_e_f2 = eval_res_f2(2:end, 4:5);
s_e_f2 = [s_e_f2 zeros(size(s_e_f2, 1), 1)];
dyn_est_err_f2 = eval_res_f2(2:end, 6:8);
u_f2 = eval_res_f2(2:end, 9:11);
angle_f2 = eval_res_f2(2:end, 12);

%% Reward plot (single plot)
figure
hold on; box on;
set(gca, 'FontSize', 14);
plot(reward, 'b-', 'LineWidth', 1.5); % Reward plot with solid blue line
legend('奖励', 'FontSize', 18);
set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
xlabel('训练次数', 'FontName', 'SimSun', 'FontSize', 18);
ylabel('奖励', 'FontName', 'SimSun', 'FontSize', 18, 'Interpreter', 'latex');
set(gcf, 'Unit', 'centimeters', 'Position', [5, 2, 25, 15]);

%% omega_e_fault_free, u_fault_free, and angle_fault_free plot (3x1 subplot)
if 0
    figure;
    
    x_label = '时间(s)';
    % omega_e_fault_free Plot (angular velocity error)
    subplot(1, 1, 1); % Create 3x1 grid, plot in 1st position
    hold on; box on;
    set(gca, 'FontSize', 14);
    plot(tspan, omega_e_fault_free(:, 1), 'r-', 'LineWidth', 1.5); % Omega_x with red solid line
    plot(tspan, omega_e_fault_free(:, 2), 'g--', 'LineWidth', 1.5); % Omega_y with green dashed line
    plot(tspan, omega_e_fault_free(:, 3), 'b-.', 'LineWidth', 1.5); % Omega_z with blue dash-dot line
    legend({'$\omega_{x}$', '$\omega_{y}$', '$\omega_{z}$'}, 'FontSize', 18, 'Interpreter', 'latex');
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
    xlabel(x_label, 'FontName', 'SimSun', 'FontSize', 18);
    ylabel('角速度(deg/s)', 'FontName', 'SimSun', 'FontSize', 18, 'Interpreter', 'latex');
    set(gcf, 'Unit', 'centimeters', 'Position', [5, 2, 25, 15]);
    
    % u_fault_free Plot (control input)
    figure;
    subplot(1, 1, 1); % Create 3x1 grid, plot in 2nd position
    hold on; box on;
    set(gca, 'FontSize', 14);
    plot(tspan, u_fault_free(:, 1), 'r-', 'LineWidth', 1.5); % Control input u_x with red solid line
    plot(tspan, u_fault_free(:, 2), 'g--', 'LineWidth', 1.5); % Control input u_y with green dashed line
    plot(tspan, u_fault_free(:, 3), 'b-.', 'LineWidth', 1.5); % Control input u_z with blue dash-dot line
    legend({'$u_x$', '$u_y$', '$u_z$'}, 'FontSize', 18, 'Interpreter', 'latex');
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
    xlabel(x_label, 'FontName', 'SimSun', 'FontSize', 18);
    ylabel('控制力矩(Nm)', 'FontName', 'SimSun', 'FontSize', 18, 'Interpreter', 'latex');
    set(gcf, 'Unit', 'centimeters', 'Position', [5, 2, 25, 15]);
    
    % Angle Plot (attitude angle_fault_free)
    figure;
    subplot(1, 1, 1); % Create 3x1 grid, plot in 3rd position
    hold on; box on;
    set(gca, 'FontSize', 14);
    plot(tspan, angle_fault_free, 'm-', 'LineWidth', 1.5); % Angle with magenta solid line
    legend('$\theta$', 'FontSize', 18);
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
    xlabel(x_label, 'FontName', 'SimSun', 'FontSize', 18);
    ylabel('对日角度(deg)', 'FontName', 'SimSun', 'FontSize', 18, 'Interpreter', 'latex');
    set(gcf, 'Unit', 'centimeters', 'Position', [5, 2, 25, 15]);
    
    % Se (sun vector error)
    figure;
    subplot(1, 1, 1); % Create 3x1 grid, plot in 3rd position
    hold on; box on;
    set(gca, 'FontSize', 14);
    plot(tspan, s_e_fault_free(:, 1), 'r-', 'LineWidth', 1.5); % Angle with magenta solid line
    plot(tspan, s_e_fault_free(:, 2), 'g--', 'LineWidth', 1.5); % Omega_y with green dashed line
    plot(tspan, s_e_fault_free(:, 3), 'b-.', 'LineWidth', 1.5); % Omega_y with green dashed line
    legend('$s_{e1}$', '$s_{e2}$', '$s_{e3}$', 'FontSize', 18);
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
    xlabel(x_label, 'FontName', 'SimSun', 'FontSize', 18);
    ylabel('太阳矢量误差', 'FontName', 'SimSun', 'FontSize', 18, 'Interpreter', 'latex');
    set(gcf, 'Unit', 'centimeters', 'Position', [5, 2, 25, 15]);
end


%% omega_e_f1, u_f1, and angle_f1 plot (3x1 subplot)
if 0
    figure;
    
    x_label = '时间(s)';
    % omega_e_f1 Plot (angular velocity error)
    subplot(1, 1, 1); % Create 3x1 grid, plot in 1st position
    hold on; box on;
    set(gca, 'FontSize', 14);
    plot(tspan, omega_e_f1(:, 1), 'r-', 'LineWidth', 1.5); % Omega_x with red solid line
    plot(tspan, omega_e_f1(:, 2), 'g--', 'LineWidth', 1.5); % Omega_y with green dashed line
    plot(tspan, omega_e_f1(:, 3), 'b-.', 'LineWidth', 1.5); % Omega_z with blue dash-dot line
    legend({'$\omega_{x}$', '$\omega_{y}$', '$\omega_{z}$'}, 'FontSize', 18, 'Interpreter', 'latex');
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
    xlabel(x_label, 'FontName', 'SimSun', 'FontSize', 18);
    ylabel('角速度(deg/s)', 'FontName', 'SimSun', 'FontSize', 18, 'Interpreter', 'latex');
    set(gcf, 'Unit', 'centimeters', 'Position', [5, 2, 25, 15]);
    
    % u_f1 Plot (control input)
    figure;
    subplot(1, 1, 1); % Create 3x1 grid, plot in 2nd position
    hold on; box on;
    set(gca, 'FontSize', 14);
    plot(tspan, u_f1(:, 1), 'r-', 'LineWidth', 1.5); % Control input u_x with red solid line
    plot(tspan, u_f1(:, 2), 'g--', 'LineWidth', 1.5); % Control input u_y with green dashed line
    plot(tspan, u_f1(:, 3), 'b-.', 'LineWidth', 1.5); % Control input u_z with blue dash-dot line
    legend({'$u_x$', '$u_y$', '$u_z$'}, 'FontSize', 18, 'Interpreter', 'latex');
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
    xlabel(x_label, 'FontName', 'SimSun', 'FontSize', 18);
    ylabel('控制力矩(Nm)', 'FontName', 'SimSun', 'FontSize', 18, 'Interpreter', 'latex');
    set(gcf, 'Unit', 'centimeters', 'Position', [5, 2, 25, 15]);
    
    % Angle Plot (attitude angle_f1)
    figure;
    subplot(1, 1, 1); % Create 3x1 grid, plot in 3rd position
    hold on; box on;
    set(gca, 'FontSize', 14);
    plot(tspan, angle_f1, 'm-', 'LineWidth', 1.5); % Angle with magenta solid line
    legend('$\theta$', 'FontSize', 18);
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
    xlabel(x_label, 'FontName', 'SimSun', 'FontSize', 18);
    ylabel('对日角度(deg)', 'FontName', 'SimSun', 'FontSize', 18, 'Interpreter', 'latex');
    set(gcf, 'Unit', 'centimeters', 'Position', [5, 2, 25, 15]);
    
    % Se (sun vector error)
    figure;
    subplot(1, 1, 1); % Create 3x1 grid, plot in 3rd position
    hold on; box on;
    set(gca, 'FontSize', 14);
    plot(tspan, s_e_f1(:, 1), 'r-', 'LineWidth', 1.5); % Angle with magenta solid line
    plot(tspan, s_e_f1(:, 2), 'g--', 'LineWidth', 1.5); % Omega_y with green dashed line
    plot(tspan, s_e_f1(:, 3), 'b-.', 'LineWidth', 1.5); % Omega_y with green dashed line
    legend('$s_{e1}$', '$s_{e2}$', '$s_{e3}$', 'FontSize', 18);
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
    xlabel(x_label, 'FontName', 'SimSun', 'FontSize', 18);
    ylabel('太阳矢量误差', 'FontName', 'SimSun', 'FontSize', 18, 'Interpreter', 'latex');
    set(gcf, 'Unit', 'centimeters', 'Position', [5, 2, 25, 15]);
end


%% omega_e_f2, u_f2, and angle_f2 plot (3x1 subplot)
if 1
    figure;
    
    x_label = '时间(s)';
    % omega_e_f2 Plot (angular velocity error)
    subplot(1, 1, 1); % Create 3x1 grid, plot in 1st position
    hold on; box on;
    set(gca, 'FontSize', 14);
    plot(tspan, omega_e_f2(:, 1), 'r-', 'LineWidth', 1.5); % Omega_x with red solid line
    plot(tspan, omega_e_f2(:, 2), 'g--', 'LineWidth', 1.5); % Omega_y with green dashed line
    plot(tspan, omega_e_f2(:, 3), 'b-.', 'LineWidth', 1.5); % Omega_z with blue dash-dot line
    legend({'$\omega_{x}$', '$\omega_{y}$', '$\omega_{z}$'}, 'FontSize', 18, 'Interpreter', 'latex');
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
    xlabel(x_label, 'FontName', 'SimSun', 'FontSize', 18);
    ylabel('角速度(deg/s)', 'FontName', 'SimSun', 'FontSize', 18, 'Interpreter', 'latex');
    set(gcf, 'Unit', 'centimeters', 'Position', [5, 2, 25, 15]);
    
    % u_f2 Plot (control input)
    figure;
    subplot(1, 1, 1); % Create 3x1 grid, plot in 2nd position
    hold on; box on;
    set(gca, 'FontSize', 14);
    plot(tspan, u_f2(:, 1), 'r-', 'LineWidth', 1.5); % Control input u_x with red solid line
    plot(tspan, u_f2(:, 2), 'g--', 'LineWidth', 1.5); % Control input u_y with green dashed line
    plot(tspan, u_f2(:, 3), 'b-.', 'LineWidth', 1.5); % Control input u_z with blue dash-dot line
    legend({'$u_x$', '$u_y$', '$u_z$'}, 'FontSize', 18, 'Interpreter', 'latex');
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
    xlabel(x_label, 'FontName', 'SimSun', 'FontSize', 18);
    ylabel('控制力矩(Nm)', 'FontName', 'SimSun', 'FontSize', 18, 'Interpreter', 'latex');
    set(gcf, 'Unit', 'centimeters', 'Position', [5, 2, 25, 15]);
    
    % Angle Plot (attitude angle_f2)
    figure;
    subplot(1, 1, 1); % Create 3x1 grid, plot in 3rd position
    hold on; box on;
    set(gca, 'FontSize', 14);
    plot(tspan, angle_f2, 'm-', 'LineWidth', 1.5); % Angle with magenta solid line
    legend('$\theta$', 'FontSize', 18);
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
    xlabel(x_label, 'FontName', 'SimSun', 'FontSize', 18);
    ylabel('对日角度(deg)', 'FontName', 'SimSun', 'FontSize', 18, 'Interpreter', 'latex');
    set(gcf, 'Unit', 'centimeters', 'Position', [5, 2, 25, 15]);
    
    % Se (sun vector error)
    figure;
    subplot(1, 1, 1); % Create 3x1 grid, plot in 3rd position
    hold on; box on;
    set(gca, 'FontSize', 14);
    plot(tspan, s_e_f2(:, 1), 'r-', 'LineWidth', 1.5); % Angle with magenta solid line
    plot(tspan, s_e_f2(:, 2), 'g--', 'LineWidth', 1.5); % Omega_y with green dashed line
    plot(tspan, s_e_f2(:, 3), 'b-.', 'LineWidth', 1.5); % Omega_y with green dashed line
    legend('$s_{e1}$', '$s_{e2}$', '$s_{e3}$', 'FontSize', 18);
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
    xlabel(x_label, 'FontName', 'SimSun', 'FontSize', 18);
    ylabel('太阳矢量误差', 'FontName', 'SimSun', 'FontSize', 18, 'Interpreter', 'latex');
    set(gcf, 'Unit', 'centimeters', 'Position', [5, 2, 25, 15]);
end


