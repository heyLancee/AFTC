import pandas as pd
import torch
import gym
import argparse
import os
import utils
import TD3
from DynamicNet import AttitudeDynamicsNN
from eval_model import eval_policy
from satellite import *


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dir", default="")
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3)
	parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=2e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=1e4, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--eval_episodes", default=10, type=int)
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
	parser.add_argument("--policy_hidden_size", default=512, type=int)  # Policy hidden size
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
	parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
	parser.add_argument("--policy_noise", default=0.2, type=float)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5, type=float)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates	
	parser.add_argument("--lr", default=3e-4, type=float)          # Learning rate
	parser.add_argument("--save_model", action="store_true", default=True)       # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--fault_mode", default=-1)		# Fault mode
	parser.add_argument("--dyn_hidden_size", default=128, type=int)  # Dynamic net hidden size
	parser.add_argument("--dyn_net_path", default="")  # Dynamic net path
	args = parser.parse_args()

	# args.policy = "TD3"
	# args.seed = 0
	# args.env = "SunPointFaultSatellite"
	# args.fault_mode = 1
	# args.dyn_net_path = "models/dynamic_net/attitude_dynamics_model.pth"

	if args.dir != "":
		file_name = f"{args.dir}/{args.policy}_{args.env}_{args.seed}"
	else:
		file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Dir: {args.dir}, Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists(f"./results/{args.dir}"):
		os.makedirs(f"./results/{args.dir}")

	if args.save_model and not os.path.exists(f"./models/{args.dir}"):
		os.makedirs(f"./models/{args.dir}")

	if args.env == "Satellite":
		env = Satellite()
	if args.env == "Satellite":
		env = Satellite()
	elif args.env == "FaultSatellite":
		env = FaultSatellite()
	elif args.env == "SunPointSatellite":
		env = SunPointSatellite()
	elif args.env == "SunPointFaultSatellite":
		env = SunPointFaultSatellite()
	else:
		env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	# state_dim append with the vars related to dyanmic net
	state_dim += TD3.STATE_APPEND_NUM
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		"lr": args.lr,
		"hidden_size": args.policy_hidden_size,
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)
	else:
		raise NotImplementedError

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		print("agent load model: ", policy_file)
		policy.load(f"./models/{policy_file}")

	dynamic_net = AttitudeDynamicsNN(args.dyn_hidden_size)
	if args.dyn_net_path != "":
		print(f"Load dynamic net: {args.dyn_net_path}")
		dynamic_net.load_model(args.dyn_net_path)

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

	state, done = env.reset(), False
	if args.fault_mode != -1:
		env.fault_mode = args.fault_mode
	state = np.concatenate((state, np.zeros(TD3.STATE_APPEND_NUM)))

	# Evaluate untrained policy
	evaluations = [eval_policy(policy, dynamic_net, args.env, env.fault_mode, args.seed, path=None)]

	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	episode_total_num = int(args.max_timesteps / (env.t_max / env.ts))

	pred_errors = []

	for t in range(int(args.max_timesteps)):
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)
		torque = np.diag(action) @ env.u_max

		# dynamic net
		net_input = np.concatenate((env.omega.flatten(), (env.C@torque).flatten()))
		pred = dynamic_net(torch.tensor(net_input, dtype=torch.float32).unsqueeze(0)).cpu().detach().numpy()

		# Perform action
		next_state, reward, done, _ = env.step(torque.reshape(-1, 1))
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		pred_error = env.omega.flatten() - pred.flatten()
		pred_errors.append(pred_error)
	
		next_state = np.concatenate((next_state.flatten(), pred_error.flatten()))
		
		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Dir: {args.dir}, Total T: {t+1}, Episode Num: {episode_num+1}/{episode_total_num}, Episode T: {episode_timesteps}, Reward: {episode_reward:.3f}")
			# Reset environment
			state, done = env.reset(), False
			state = np.concatenate((state, np.zeros(TD3.STATE_APPEND_NUM)))
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1

			# plt.plot(np.array(pred_errors))
			# plt.show()

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			eval_reward = 0
			for _ in range(args.eval_episodes):
				eval_reward += eval_policy(policy, dynamic_net, args.env, env.fault_mode, args.seed, path=None)
			evaluations.append(eval_reward / args.eval_episodes)
			pd.DataFrame(evaluations).to_csv(f"./results/{file_name}.csv")
			if args.save_model:
				policy.save(f"./models/{file_name}")
	