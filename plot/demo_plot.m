% 画图 Demo
% Author: Wenhan zhang
% Date: 2022, Oct 10

clc;
clear;
close all;

%%
% 动力学网络相关
Time = 200;
Ts = 0.1;
tspan = 0:Ts:Time;
pred_actu_faultfree = readmatrix ('../results/dyn_net/eval_preds_actuals_Satellite_0.csv');
pred_actu_f1 = readmatrix ('../results/dyn_net/eval_preds_actuals_FaultSatellite_1.csv');
pred_actu_f2 = readmatrix ('../results/dyn_net/eval_preds_actuals_FaultSatellite_2.csv');

pred_omega_x_faultfree = pred_actu_faultfree(:, 1);
pred_omega_y_faultfree = pred_actu_faultfree(:, 2);
pred_omega_z_faultfree = pred_actu_faultfree(:, 3);
actu_omega_x_faultfree = pred_actu_faultfree(:, 4);
actu_omega_y_faultfree = pred_actu_faultfree(:, 5);
actu_omega_z_faultfree = pred_actu_faultfree(:, 6);

pred_omega_x_f1 = pred_actu_f1(:, 1);
pred_omega_y_f1 = pred_actu_f1(:, 2);
pred_omega_z_f1 = pred_actu_f1(:, 3);
actu_omega_x_f1 = pred_actu_f1(:, 4);
actu_omega_y_f1 = pred_actu_f1(:, 5);
actu_omega_z_f1 = pred_actu_f1(:, 6);

pred_omega_x_f2 = pred_actu_f2(:, 1);
pred_omega_y_f2 = pred_actu_f2(:, 2);
pred_omega_z_f2 = pred_actu_f2(:, 3);
actu_omega_x_f2 = pred_actu_f2(:, 4);
actu_omega_y_f2 = pred_actu_f2(:, 5);
actu_omega_z_f2 = pred_actu_f2(:, 6);

err_faultfree = readmatrix ('../results/dyn_net/eval_error_Satellite_0.csv');
err_f1 = readmatrix ('../results/dyn_net/eval_error_FaultSatellite_1.csv');
err_f2 = readmatrix ('../results/dyn_net/eval_error_FaultSatellite_2.csv');

err_omega_x_faultfree = err_faultfree(:, 1);
err_omega_y_faultfree = err_faultfree(:, 2);
err_omega_z_faultfree = err_faultfree(:, 3);

err_omega_x_f1 = err_f1(:, 1);
err_omega_y_f1 = err_f1(:, 2);
err_omega_z_f1 = err_f1(:, 3);

err_omega_x_f2 = err_f2(:, 1);
err_omega_y_f2 = err_f2(:, 2);
err_omega_z_f2 = err_f2(:, 3);

x_label = 'Time(s)';

%% subplots
% fault free pred and actual
figure
subplot(3, 1, 1);
hold on; box on;
set(gca, 'FontSize', 14);
plot(tspan, pred_omega_x_faultfree,'r-','linewidth',1.5);
plot(tspan, actu_omega_x_faultfree,'b--','linewidth',1.5);

legend('Dynamic Network Output','Actual Output', 'fontsize',18);
set(legend,'Interpreter','latex','Orientation','horizontal','box','off');
set(gcf,'unit','centimeters','position',[5,2,25,15]);
y_label = {'(rad/s)'};
xlabel(x_label,'fontname','Times New Roman','fontsize',18);
ylabel(y_label,'fontname','Times New Roman','fontsize',18,'Interpreter','latex');
set(gcf,'unit','centimeters','position',[5,2,25,15]);

subplot(3, 1, 2);
hold on; box on;
set(gca, 'FontSize', 14);
plot(tspan, pred_omega_y_faultfree,'r-','linewidth',1.5);
plot(tspan, actu_omega_y_faultfree,'b--','linewidth',1.5);

legend('Dynamic Network Output','Actual Output', 'fontsize',18);
set(legend,'Interpreter','latex','Orientation','horizontal','box','off');
set(gcf,'unit','centimeters','position',[5,2,25,15]);
y_label = {'(rad/s)'};
xlabel(x_label,'fontname','Times New Roman','fontsize',18);
ylabel(y_label,'fontname','Times New Roman','fontsize',18,'Interpreter','latex');
set(gcf,'unit','centimeters','position',[5,2,25,15]);

subplot(3, 1, 3);
hold on; box on;
set(gca, 'FontSize', 14);
plot(tspan, pred_omega_z_faultfree,'r-','linewidth',1.5);
plot(tspan, actu_omega_z_faultfree,'b--','linewidth',1.5);

legend('Dynamic Network Output','Actual Output', 'fontsize',18);
set(legend,'Interpreter','latex','Orientation','horizontal','box','off');
set(gcf,'unit','centimeters','position',[5,2,25,15]);
y_label = {'(rad/s)'};
xlabel(x_label,'fontname','Times New Roman','fontsize',18);
ylabel(y_label,'fontname','Times New Roman','fontsize',18,'Interpreter','latex');
set(gcf,'unit','centimeters','position',[5,2,25,15]);

% fault free error
figure;
subplot(3, 1, 1);
hold on; box on;
set(gca, 'FontSize', 14);
plot(tspan, err_omega_x_faultfree,'r-','linewidth',1.5);

legend('Estimation Error', 'fontsize',18);
set(legend,'Interpreter','latex','Orientation','horizontal','box','off');
set(gcf,'unit','centimeters','position',[5,2,25,15]);
y_label = {'(rad/s)'};
xlabel(x_label,'fontname','Times New Roman','fontsize',18);
ylabel(y_label,'fontname','Times New Roman','fontsize',18,'Interpreter','latex');
set(gcf,'unit','centimeters','position',[5,2,25,15]);

subplot(3, 1, 2);
hold on; box on;
set(gca, 'FontSize', 14);
plot(tspan, err_omega_y_faultfree,'r-','linewidth',1.5);

legend('Estimation Error', 'fontsize',18);
set(legend,'Interpreter','latex','Orientation','horizontal','box','off');
set(gcf,'unit','centimeters','position',[5,2,25,15]);
y_label = {'(rad/s)'};
xlabel(x_label,'fontname','Times New Roman','fontsize',18);
ylabel(y_label,'fontname','Times New Roman','fontsize',18,'Interpreter','latex');
set(gcf,'unit','centimeters','position',[5,2,25,15]);

subplot(3, 1, 3);
hold on; box on;
set(gca, 'FontSize', 14);
plot(tspan, err_omega_z_faultfree,'r-','linewidth',1.5);

legend('Estimation Error', 'fontsize',18);
set(legend,'Interpreter','latex','Orientation','horizontal','box','off');
set(gcf,'unit','centimeters','position',[5,2,25,15]);
y_label = {'(rad/s)'};
xlabel(x_label,'fontname','Times New Roman','fontsize',18);
ylabel(y_label,'fontname','Times New Roman','fontsize',18,'Interpreter','latex');
set(gcf,'unit','centimeters','position',[5,2,25,15]);

%% f1
% f1 pred and actual
figure
subplot(3, 1, 1);
hold on; box on;
set(gca, 'FontSize', 14);
plot(tspan, pred_omega_x_f1,'r-','linewidth',1.5);
plot(tspan, actu_omega_x_f1,'b--','linewidth',1.5);

legend('Dynamic Network Output','Actual Output', 'fontsize',18);
set(legend,'Interpreter','latex','Orientation','horizontal','box','off');
set(gcf,'unit','centimeters','position',[5,2,25,15]);
y_label = {'(rad/s)'};
xlabel(x_label,'fontname','Times New Roman','fontsize',18);
ylabel(y_label,'fontname','Times New Roman','fontsize',18,'Interpreter','latex');
set(gcf,'unit','centimeters','position',[5,2,25,15]);

subplot(3, 1, 2);
hold on; box on;
set(gca, 'FontSize', 14);
plot(tspan, pred_omega_y_f1,'r-','linewidth',1.5);
plot(tspan, actu_omega_y_f1,'b--','linewidth',1.5);

legend('Dynamic Network Output','Actual Output', 'fontsize',18);
set(legend,'Interpreter','latex','Orientation','horizontal','box','off');
set(gcf,'unit','centimeters','position',[5,2,25,15]);
y_label = {'(rad/s)'};
xlabel(x_label,'fontname','Times New Roman','fontsize',18);
ylabel(y_label,'fontname','Times New Roman','fontsize',18,'Interpreter','latex');
set(gcf,'unit','centimeters','position',[5,2,25,15]);

subplot(3, 1, 3);
hold on; box on;
set(gca, 'FontSize', 14);
plot(tspan, pred_omega_z_f1,'r-','linewidth',1.5);
plot(tspan, actu_omega_z_f1,'b--','linewidth',1.5);

legend('Dynamic Network Output','Actual Output', 'fontsize',18);
set(legend,'Interpreter','latex','Orientation','horizontal','box','off');
set(gcf,'unit','centimeters','position',[5,2,25,15]);
y_label = {'(rad/s)'};
xlabel(x_label,'fontname','Times New Roman','fontsize',18);
ylabel(y_label,'fontname','Times New Roman','fontsize',18,'Interpreter','latex');
set(gcf,'unit','centimeters','position',[5,2,25,15]);

% f1 error
figure;
subplot(3, 1, 1);
hold on; box on;
set(gca, 'FontSize', 14);
plot(tspan, err_omega_x_f1,'r-','linewidth',1.5);

legend('Estimation Error', 'fontsize',18);
set(legend,'Interpreter','latex','Orientation','horizontal','box','off');
set(gcf,'unit','centimeters','position',[5,2,25,15]);
y_label = {'(rad/s)'};
xlabel(x_label,'fontname','Times New Roman','fontsize',18);
ylabel(y_label,'fontname','Times New Roman','fontsize',18,'Interpreter','latex');
set(gcf,'unit','centimeters','position',[5,2,25,15]);

subplot(3, 1, 2);
hold on; box on;
set(gca, 'FontSize', 14);
plot(tspan, err_omega_y_f1,'r-','linewidth',1.5);

legend('Estimation Error', 'fontsize',18);
set(legend,'Interpreter','latex','Orientation','horizontal','box','off');
set(gcf,'unit','centimeters','position',[5,2,25,15]);
y_label = {'(rad/s)'};
xlabel(x_label,'fontname','Times New Roman','fontsize',18);
ylabel(y_label,'fontname','Times New Roman','fontsize',18,'Interpreter','latex');
set(gcf,'unit','centimeters','position',[5,2,25,15]);

subplot(3, 1, 3);
hold on; box on;
set(gca, 'FontSize', 14);
plot(tspan, err_omega_z_f1,'r-','linewidth',1.5);

legend('Estimation Error', 'fontsize',18);
set(legend,'Interpreter','latex','Orientation','horizontal','box','off');
set(gcf,'unit','centimeters','position',[5,2,25,15]);
y_label = {'(rad/s)'};
xlabel(x_label,'fontname','Times New Roman','fontsize',18);
ylabel(y_label,'fontname','Times New Roman','fontsize',18,'Interpreter','latex');
set(gcf,'unit','centimeters','position',[5,2,25,15]);

%% f2
% f2 pred and actual
figure
subplot(3, 1, 1);
hold on; box on;
set(gca, 'FontSize', 14);
plot(tspan, pred_omega_x_f2,'r-','linewidth',1.5);
plot(tspan, actu_omega_x_f2,'b--','linewidth',1.5);

legend('Dynamic Network Output','Actual Output', 'fontsize',18);
set(legend,'Interpreter','latex','Orientation','horizontal','box','off');
set(gcf,'unit','centimeters','position',[5,2,25,15]);
y_label = {'(rad/s)'};
xlabel(x_label,'fontname','Times New Roman','fontsize',18);
ylabel(y_label,'fontname','Times New Roman','fontsize',18,'Interpreter','latex');
set(gcf,'unit','centimeters','position',[5,2,25,15]);

subplot(3, 1, 2);
hold on; box on;
set(gca, 'FontSize', 14);
plot(tspan, pred_omega_y_f2,'r-','linewidth',1.5);
plot(tspan, actu_omega_y_f2,'b--','linewidth',1.5);

legend('Dynamic Network Output','Actual Output', 'fontsize',18);
set(legend,'Interpreter','latex','Orientation','horizontal','box','off');
set(gcf,'unit','centimeters','position',[5,2,25,15]);
y_label = {'(rad/s)'};
xlabel(x_label,'fontname','Times New Roman','fontsize',18);
ylabel(y_label,'fontname','Times New Roman','fontsize',18,'Interpreter','latex');
set(gcf,'unit','centimeters','position',[5,2,25,15]);

subplot(3, 1, 3);
hold on; box on;
set(gca, 'FontSize', 14);
plot(tspan, pred_omega_z_f2,'r-','linewidth',1.5);
plot(tspan, actu_omega_z_f2,'b--','linewidth',1.5);

legend('Dynamic Network Output','Actual Output', 'fontsize',18);
set(legend,'Interpreter','latex','Orientation','horizontal','box','off');
set(gcf,'unit','centimeters','position',[5,2,25,15]);
y_label = {'(rad/s)'};
xlabel(x_label,'fontname','Times New Roman','fontsize',18);
ylabel(y_label,'fontname','Times New Roman','fontsize',18,'Interpreter','latex');
set(gcf,'unit','centimeters','position',[5,2,25,15]);

% f2 error
figure;
subplot(3, 1, 1);
hold on; box on;
set(gca, 'FontSize', 14);
plot(tspan, err_omega_x_f2,'r-','linewidth',1.5);

legend('Estimation Error', 'fontsize',18);
set(legend,'Interpreter','latex','Orientation','horizontal','box','off');
set(gcf,'unit','centimeters','position',[5,2,25,15]);
y_label = {'(rad/s)'};
xlabel(x_label,'fontname','Times New Roman','fontsize',18);
ylabel(y_label,'fontname','Times New Roman','fontsize',18,'Interpreter','latex');
set(gcf,'unit','centimeters','position',[5,2,25,15]);

subplot(3, 1, 2);
hold on; box on;
set(gca, 'FontSize', 14);
plot(tspan, err_omega_y_f2,'r-','linewidth',1.5);

legend('Estimation Error', 'fontsize',18);
set(legend,'Interpreter','latex','Orientation','horizontal','box','off');
set(gcf,'unit','centimeters','position',[5,2,25,15]);
y_label = {'(rad/s)'};
xlabel(x_label,'fontname','Times New Roman','fontsize',18);
ylabel(y_label,'fontname','Times New Roman','fontsize',18,'Interpreter','latex');
set(gcf,'unit','centimeters','position',[5,2,25,15]);

subplot(3, 1, 3);
hold on; box on;
set(gca, 'FontSize', 14);
plot(tspan, err_omega_z_f2,'r-','linewidth',1.5);

legend('Estimation Error', 'fontsize',18);
set(legend,'Interpreter','latex','Orientation','horizontal','box','off');
set(gcf,'unit','centimeters','position',[5,2,25,15]);
y_label = {'(rad/s)'};
xlabel(x_label,'fontname','Times New Roman','fontsize',18);
ylabel(y_label,'fontname','Times New Roman','fontsize',18,'Interpreter','latex');
set(gcf,'unit','centimeters','position',[5,2,25,15]);








