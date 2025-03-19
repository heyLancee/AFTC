% 画图 Demo
% Author: Wenhan zhang
% Date: 2022, Oct 10

clc;
clear;
close all;

%%
% reward
reward1 = readmatrix('../results/u_max_005_2/TD3_SunPointFaultSatellite_5.csv');
reward2 = readmatrix('../results/u_max_005/TD3_SunPointFaultSatellite_3.csv');
reward3 = readmatrix('../results/u_max_005_3/TD3_SunPointFaultSatellite_1.csv');

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

% unknown_fault
eval_res_unknown_fault = readmatrix ('../results/u_max_005/eval_res_unknown_fault.csv');
omega_e_unknown_fault = eval_res_unknown_fault(2:end, 1:3) * 180 / pi;
s_e_unknown_fault = eval_res_unknown_fault(2:end, 4:5);
s_e_unknown_fault = [s_e_unknown_fault zeros(size(s_e_unknown_fault, 1), 1)];
dyn_est_err_unknown_fault = eval_res_unknown_fault(2:end, 6:8);
u_unknown_fault = eval_res_unknown_fault(2:end, 9:11);
angle_unknown_fault = eval_res_unknown_fault(2:end, 12);

%% Reward plot (single plot)
x = 1:length(reward1);
all_rewards = [reward1; reward2; reward3]';
mean_rewards = mean(all_rewards, 2);
std_rewards = std(all_rewards, 0, 2);
CI = 1.96 * std_rewards / sqrt(size(all_rewards, 2)); % 计算每个时间步的置信区间

figure;
hold on;
set(gca, 'FontSize', 14);
set(gcf, 'Unit', 'centimeters', 'Position', [5, 2, 25, 20]);
fill([x, fliplr(x)], [mean_rewards + CI; flipud(mean_rewards - CI)], 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
plot(x, mean_rewards, 'b', 'LineWidth', 1.5);
xlabel('训练次数', 'FontName', 'SimSun', 'FontSize', 18);
ylabel('奖励', 'FontName', 'SimSun', 'FontSize', 18, 'Interpreter', 'latex');
legend('95%置信区间', 'Location', 'Best', 'FontSize', 18);
set(gca, 'box', 'off'); % 关闭默认的四边框
set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
hold off;

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
    set(gca, 'box', 'off'); % 关闭默认的四边框
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
    
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
    set(gca, 'box', 'off'); % 关闭默认的四边框
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');

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
    set(gca, 'box', 'off'); % 关闭默认的四边框
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
    
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
    set(gca, 'box', 'off'); % 关闭默认的四边框
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
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
    set(gca, 'box', 'off'); % 关闭默认的四边框
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
    
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
    set(gca, 'box', 'off'); % 关闭默认的四边框
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
    
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
    set(gca, 'box', 'off'); % 关闭默认的四边框
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');

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
    set(gca, 'box', 'off'); % 关闭默认的四边框
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
end


%% omega_e_f2, u_f2, and angle_f2 plot (3x1 subplot)
if 0
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
    set(gca, 'box', 'off'); % 关闭默认的四边框
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
    
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
    set(gca, 'box', 'off'); % 关闭默认的四边框
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
    
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
    set(gca, 'box', 'off'); % 关闭默认的四边框
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
    
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
    set(gca, 'box', 'off'); % 关闭默认的四边框
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
end

%% omega_e_unknown_fault, u_unknown_fault, and angle_unknown_fault plot (3x1 subplot)
if 1
    figure;
    
    x_label = '时间(s)';
    % omega_e_unknown_fault Plot (angular velocity error)
    subplot(1, 1, 1); % Create 3x1 grid, plot in 1st position
    hold on; box on;
    set(gca, 'FontSize', 14);
    plot(tspan, omega_e_unknown_fault(:, 1), 'r-', 'LineWidth', 1.5); % Omega_x with red solid line
    plot(tspan, omega_e_unknown_fault(:, 2), 'g--', 'LineWidth', 1.5); % Omega_y with green dashed line
    plot(tspan, omega_e_unknown_fault(:, 3), 'b-.', 'LineWidth', 1.5); % Omega_z with blue dash-dot line
    legend({'$\omega_{x}$', '$\omega_{y}$', '$\omega_{z}$'}, 'FontSize', 18, 'Interpreter', 'latex');
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
    xlabel(x_label, 'FontName', 'SimSun', 'FontSize', 18);
    ylabel('角速度(deg/s)', 'FontName', 'SimSun', 'FontSize', 18, 'Interpreter', 'latex');
    set(gcf, 'Unit', 'centimeters', 'Position', [5, 2, 25, 15]);
    set(gca, 'box', 'off'); % 关闭默认的四边框
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
    
    % u_unknown_fault Plot (control input)
    figure;
    subplot(1, 1, 1); % Create 3x1 grid, plot in 2nd position
    hold on; box on;
    set(gca, 'FontSize', 14);
    plot(tspan, u_unknown_fault(:, 1), 'r-', 'LineWidth', 1.5); % Control input u_x with red solid line
    plot(tspan, u_unknown_fault(:, 2), 'g--', 'LineWidth', 1.5); % Control input u_y with green dashed line
    plot(tspan, u_unknown_fault(:, 3), 'b-.', 'LineWidth', 1.5); % Control input u_z with blue dash-dot line
    legend({'$u_x$', '$u_y$', '$u_z$'}, 'FontSize', 18, 'Interpreter', 'latex');
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
    xlabel(x_label, 'FontName', 'SimSun', 'FontSize', 18);
    ylabel('控制力矩(Nm)', 'FontName', 'SimSun', 'FontSize', 18, 'Interpreter', 'latex');
    set(gcf, 'Unit', 'centimeters', 'Position', [5, 2, 25, 15]);
    set(gca, 'box', 'off'); % 关闭默认的四边框
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
    
    % Angle Plot (attitude angle_unknown_fault)
    figure;
    subplot(1, 1, 1); % Create 3x1 grid, plot in 3rd position
    hold on; box on;
    set(gca, 'FontSize', 14);
    plot(tspan, angle_unknown_fault, 'm-', 'LineWidth', 1.5); % Angle with magenta solid line
    legend('$\theta$', 'FontSize', 18);
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
    xlabel(x_label, 'FontName', 'SimSun', 'FontSize', 18);
    ylabel('对日角度(deg)', 'FontName', 'SimSun', 'FontSize', 18, 'Interpreter', 'latex');
    set(gcf, 'Unit', 'centimeters', 'Position', [5, 2, 25, 15]);
    set(gca, 'box', 'off'); % 关闭默认的四边框
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
    
    % Se (sun vector error)
    figure;
    subplot(1, 1, 1); % Create 3x1 grid, plot in 3rd position
    hold on; box on;
    set(gca, 'FontSize', 14);
    plot(tspan, s_e_unknown_fault(:, 1), 'r-', 'LineWidth', 1.5); % Angle with magenta solid line
    plot(tspan, s_e_unknown_fault(:, 2), 'g--', 'LineWidth', 1.5); % Omega_y with green dashed line
    plot(tspan, s_e_unknown_fault(:, 3), 'b-.', 'LineWidth', 1.5); % Omega_y with green dashed line
    legend('$s_{e1}$', '$s_{e2}$', '$s_{e3}$', 'FontSize', 18);
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
    xlabel(x_label, 'FontName', 'SimSun', 'FontSize', 18);
    ylabel('太阳矢量误差', 'FontName', 'SimSun', 'FontSize', 18, 'Interpreter', 'latex');
    set(gcf, 'Unit', 'centimeters', 'Position', [5, 2, 25, 15]);
    set(gca, 'box', 'off'); % 关闭默认的四边框
    set(legend, 'Interpreter', 'latex', 'Orientation', 'horizontal', 'Box', 'off');
end
