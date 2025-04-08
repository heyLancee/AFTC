import os
import sys

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
root_path = os.path.dirname(parent_dir)
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import List
import argparse

from src.satellite import *
import src.td3 as td3

INPUT_NUM = 6
OUTPUT_NUM = td3.STATE_APPEND_NUM

torch.set_printoptions(precision=8)


class AttitudeDynamicsNN(nn.Module):
    def __init__(self, hidden_size: List[int]):
        super(AttitudeDynamicsNN, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.fc1 = nn.Linear(INPUT_NUM, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], OUTPUT_NUM)

    @staticmethod
    def normalize_quaternion(q):
        q_norm = torch.norm(q, p=2, dim=1, keepdim=True)
        q_normalized = q / q_norm
        return q_normalized

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_model(self, X_train, y_train, X_test, y_test, num_epochs=1000, batch_size=32, learning_rate=0.001,
                    model_path="dynNet.pth", data_path="data.csv"):
        self.train()

        self.to(self.device)
        # 将数据转换为PyTorch张量
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(self.device)

        # 定义损失函数和优化器
        criterion = nn.MSELoss()  # 使用均方误差作为损失函数
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # 训练过程中逐渐降低学习率
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

        eval_loss = []

        # 训练模型
        for epoch in range(num_epochs):
            # 打乱训练数据
            permutation = torch.randperm(X_train.size(0))
            X_train = X_train[permutation]
            y_train = y_train[permutation]
            # 分批训练
            for i in range(0, X_train.size(0), batch_size):
                inputs = X_train[i:i + batch_size]
                targets = y_train[i:i + batch_size]
                # 前向传播
                omega_next = self.forward(inputs)
                outputs = omega_next
                # 计算损失
                loss = criterion(outputs, targets)
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 降低学习率
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                # test
                y_pred = self.forward(X_test)
                eval_loss.append(criterion(y_pred, y_test).item())

                print(f"Epoch [{epoch + 1}/{num_epochs}], Eval loss: {eval_loss[-1]:.8f}")
                self.save_model(model_path)
                pd.DataFrame(eval_loss, columns=["eval_loss"]).to_csv(data_path, index=False)

    def predict(self, x_test):
        self.eval()
        with torch.no_grad():
            q, omega = self.forward(x_test)
        return q, omega

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="Satellite", type=str)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--data_path", default="../sample_data/data.csv", type=str)
    parser.add_argument("--model_save_path", default="./models/dynamic_net/attitude_dynamics_model_4_step.pth", type=str)
    parser.add_argument("--data_save_path", default="./models/dynamic_net/eval_loss.csv", type=str)
    parser.add_argument("--model_load_path", default="", type=str)
    parser.add_argument("--test_size_per", default=0.1, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_epochs", default=500, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--save_model", action="store_true", default=False)
    parser.add_argument("--fault_mode", default=0, type=int)

    args = parser.parse_args()

    env_name = args.env_name
    hidden_size = args.hidden_size
    data_path = args.data_path
    model_save_path = args.model_save_path
    data_save_path = args.data_save_path
    model_load_path = args.model_load_path
    test_size_per = args.test_size_per
    seed = args.seed
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    save_model = args.save_model
    fault_mode = args.fault_mode

    env_name = "FaultSatellite"
    hidden_size = [64, 128]
    data_path = "../sample_data/data.csv"
    save_model = True
    fault_mode = 2
    model_load_path = r"dynamic_net/attitude_dynamics_model.pth"

    # 打印一些log
    print("---------------------------------------")
    print("env_name: ", env_name)
    print("hidden_size: ", hidden_size)
    print("data_path: ", data_path)
    print("model_save_path: ", model_save_path)
    print("model_load_path: ", model_load_path)
    print("test_size_per: ", test_size_per)
    print("seed: ", seed)
    print("num_epochs: ", num_epochs)
    print("batch_size: ", batch_size)
    print("lr: ", lr)
    print("save_model: ", save_model)
    print("---------------------------------------")

    model = AttitudeDynamicsNN(hidden_size=hidden_size)

    if model_load_path != "":
        print("load model: ", model_load_path)
        model.load_model(f"../models/{model_load_path}")

    data = pd.read_csv(data_path)
    x = data.iloc[:, :INPUT_NUM].values
    y = data.iloc[:, INPUT_NUM:].values

    # split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size_per, random_state=seed)

    if save_model:
        # 训练模型
        model.train_model(x_train, y_train, x_test, y_test, num_epochs=num_epochs, batch_size=batch_size,
                          learning_rate=lr, model_path=model_save_path, data_path=data_save_path)
        # 保存模型
        model.save_model(model_save_path)
