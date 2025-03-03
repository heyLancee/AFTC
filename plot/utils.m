clc;
clear;
close all;

% reward
reward = readmatrix ('../results/u_max_005_2/TD3_SunPointFaultSatellite_5.csv');
reward = reward(2:end, 2);

% eval
Time = 200;
Ts = 0.1;
tspan = Ts:Ts:Time;

% fault free
eval_res_fault_free = readmatrix ('../results/u_max_005/eval_res_fault_free.csv');
omega_fault_free = eval_res_fault_free(2:end, 1:3) * 180 / pi;
omega_fault_free(:, 3) = omega_fault_free(:, 3) - 0.2;
s_e_fault_free = eval_res_fault_free(2:end, 4:5);
s_e_fault_free = [s_e_fault_free zeros(size(s_e_fault_free, 1), 1)];
dyn_est_err_fault_free = eval_res_fault_free(2:end, 6:8);
u_fault_free = eval_res_fault_free(2:end, 9:11);
angle_fault_free = eval_res_fault_free(2:end, 12);

% f1
eval_res_f1 = readmatrix ('../results/u_max_005/eval_res_f1.csv');
omega_f1 = eval_res_f1(2:end, 1:3) * 180 / pi;
omega_f1(:, 3) = omega_f1(:, 3) - 0.2;
s_e_f1 = eval_res_f1(2:end, 4:5);
s_e_f1 = [s_e_f1 zeros(size(s_e_f1, 1), 1)];
dyn_est_err_f1 = eval_res_f1(2:end, 6:8);
u_f1 = eval_res_f1(2:end, 9:11);
angle_f1 = eval_res_f1(2:end, 12);

% f2
eval_res_f2 = readmatrix ('../results/u_max_005/eval_res_f2.csv');
omega_f2 = eval_res_f2(2:end, 1:3) * 180 / pi;
omega_f2(:, 3) = omega_f2(:, 3) - 0.2;
s_e_f2 = eval_res_f2(2:end, 4:5);
s_e_f2 = [s_e_f2 zeros(size(s_e_f2, 1), 1)];
dyn_est_err_f2 = eval_res_f2(2:end, 6:8);
u_f2 = eval_res_f2(2:end, 9:11);
angle_f2 = eval_res_f2(2:end, 12);

% unknown_fault
eval_res_unknown_fault = readmatrix ('../results/u_max_005/eval_res_unknown_fault.csv');
omega_unknown_fault = eval_res_unknown_fault(2:end, 1:3) * 180 / pi;
omega_unknown_fault(:, 3) = omega_unknown_fault(:, 3) - 0.2;
s_e_unknown_fault = eval_res_unknown_fault(2:end, 4:5);
s_e_unknown_fault = [s_e_unknown_fault zeros(size(s_e_unknown_fault, 1), 1)];
dyn_est_err_unknown_fault = eval_res_unknown_fault(2:end, 6:8);
u_unknown_fault = eval_res_unknown_fault(2:end, 9:11);
angle_unknown_fault = eval_res_unknown_fault(2:end, 12);

%%
omega_th = 0.05;  % deg/s
angle_th = 3;  % deg

% fault free
omega_min_time_fault_free = inf;
for i = size(omega_fault_free, 1):-1:1
    if all(abs(omega_fault_free(i, :)) <= omega_th)  % 检查当前行是否所有元素都小于等于阈值
        omega_min_time_fault_free = i;  % 如果当前行符合条件，记录该时间步
    else
        break;
    end
end
angle_min_time_fault_free = inf;
for i = size(angle_fault_free, 1):-1:1
    if all(abs(angle_fault_free(i, :)) <= angle_th)  % 检查当前行是否所有元素都小于等于阈值
        angle_min_time_fault_free = i;  % 如果当前行符合条件，记录该时间步
    else
        break;
    end
end

% f1
omega_min_time_f1 = inf;
for i = size(omega_f1, 1):-1:1
    if all(abs(omega_f1(i, :)) <= omega_th)  % 检查当前行是否所有元素都小于等于阈值
        omega_min_time_f1 = i;  % 如果当前行符合条件，记录该时间步
    else
        break;
    end
end
angle_min_time_f1 = inf;
for i = size(angle_f1, 1):-1:1
    if all(abs(angle_f1(i, :)) <= angle_th)  % 检查当前行是否所有元素都小于等于阈值
        angle_min_time_f1 = i;  % 如果当前行符合条件，记录该时间步
    else
        break;
    end
end

% f2
omega_min_time_f2 = inf;
for i = size(omega_f2, 1):-1:1
    if all(abs(omega_f2(i, :)) <= omega_th)  % 检查当前行是否所有元素都小于等于阈值
        omega_min_time_f2 = i;  % 如果当前行符合条件，记录该时间步
    else
        break;
    end
end
angle_min_time_f2 = inf;
for i = size(angle_f2, 1):-1:1
    if all(abs(angle_f2(i, :)) <= angle_th)  % 检查当前行是否所有元素都小于等于阈值
        angle_min_time_f2 = i;  % 如果当前行符合条件，记录该时间步
    else
        break;
    end
end

% unknown fault
omega_min_time_unknown_fault = inf;
for i = size(omega_unknown_fault, 1):-1:1
    if all(abs(omega_unknown_fault(i, :)) <= omega_th)  % 检查当前行是否所有元素都小于等于阈值
        omega_min_time_unknown_fault = i;  % 如果当前行符合条件，记录该时间步
    else
        break;
    end
end
angle_min_time_unknown_fault = inf;
for i = size(angle_unknown_fault, 1):-1:1
    if all(abs(angle_unknown_fault(i, :)) <= angle_th)  % 检查当前行是否所有元素都小于等于阈值
        angle_min_time_unknown_fault = i;  % 如果当前行符合条件，记录该时间步
    else
        break;
    end
end

% 显示结果
disp(['无故障情况下，角速度收敛最小时间为: ', num2str(omega_min_time_fault_free*Ts)]);
disp(['无故障情况下，对日角度收敛最小时间为: ', num2str(angle_min_time_fault_free*Ts)]);

disp(['故障1情况下，角速度收敛最小时间为: ', num2str(omega_min_time_f1*Ts)]);
disp(['故障1情况下，对日角度收敛最小时间为: ', num2str(angle_min_time_f1*Ts)]);

disp(['故障2情况下，角速度收敛最小时间为: ', num2str(omega_min_time_f2*Ts)]);
disp(['故障2情况下，对日角度收敛最小时间为: ', num2str(angle_min_time_f2*Ts)]);

disp(['未知故障情况下，角速度收敛最小时间为: ', num2str(omega_min_time_unknown_fault*Ts)]);
disp(['未知故障情况下，对日角度收敛最小时间为: ', num2str(angle_min_time_unknown_fault*Ts)]);
