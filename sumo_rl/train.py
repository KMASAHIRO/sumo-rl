import os
import torch
import numpy as np
import pandas as pd
import traci

from .env import SumoEnvironment
from .module import Agent

# 学習させる関数
def train_agent(
    net_file='/home/kato/traffic_light_control/sumo-rl/nets/2x2grid/2x2.net.xml',
    route_file='/home/kato/traffic_light_control/sumo-rl/nets/2x2grid/2x2.rou.xml',
    model_save_path=None, episode_per_learn=20, max_steps=300, episodes=10000, begin_seconds=0.0, 
    delta_time=1, yellow_time=2, min_green=5, reward_type="waiting_time", num_layers=1, 
    num_hidden_units=512, lr=3e-5, decay_rate=0.01, temperature=1.0, noise=0.0, encoder_type="fc", 
    lstm_len=5, embedding_num=5, embedding_decay=0.99, eps=1e-5, beta=0.25, reward_csv=None, 
    loss_csv=None, device="cpu", seed="random", logger=None):
    
    if reward_csv is not None:
        csv_dir = "./" + reward_csv.replace(".csv","") + "/outputs"
    else:
        csv_dir = None
    env = SumoEnvironment(net_file=net_file,
                          route_file=route_file,
                          save_state_dir=None,
                          out_csv_name=csv_dir, test=False, use_gui=False, 
                          begin_seconds=begin_seconds, delta_time=delta_time, 
                          yellow_time=yellow_time, min_green=min_green, 
                          reward_type=reward_type, single_agent=False, seed=seed)

    reset_obs = env.reset()
    traffic_light_ids = env.ts_ids
    num_states = 0
    for obs_t in reset_obs.values():
        num_states += (len(obs_t)+1) * 2
    num_actions = list()
    for id in traffic_light_ids:
        num_actions.append(env.traffic_signals[id].num_green_phases)
    
    agent = Agent(
        num_states=num_states, num_traffic_lights=len(traffic_light_ids), num_actions=num_actions, 
        num_layers=num_layers, num_hidden_units=num_hidden_units, temperature=temperature, noise=noise, 
        encoder_type=encoder_type, lr=lr, decay_rate=decay_rate, embedding_num=embedding_num, 
        embedding_decay=embedding_decay, eps=eps, beta=beta, is_train=True, device=device)
    
    steps = 0
    steps_per_learn = max_steps*episode_per_learn

    loss_list = list()
    best_reward_mean = float("-inf")
    current_reward_sum = 0
    for i in range(episodes):
        if encoder_type == "lstm":
            obs_seq = list()
        if i!= 0:
            _ = env.reset()
        
        if traffic_light_ids != env.ts_ids and logger is not None:
            logger.debug(f"right ts_ids: {traffic_light_ids}, current ts_ids: {env.ts_ids}")
        
        action = dict()
        for j in range(len(traffic_light_ids)):
            action[traffic_light_ids[j]] = None
        _ = env.step(action)

        state = env.step(action)
        obs = np.concatenate(list(state[0].values()))
        reward = np.asarray(list(state[1].values())) / float(delta_time)
        reward = np.clip(reward, -1.0, 1.0)
        if encoder_type == "lstm":
            obs_seq.append(np.concatenate((obs, reward), axis=0))
        else:
            prev_obs = np.zeros(len(obs), dtype=np.float32)
            prev_reward = [0.0 for j in range(len(reward))]
            input_obs = np.concatenate(
                (obs, reward, prev_obs, prev_reward),
                axis=0)
            prev_obs = obs
            prev_reward = reward
        end = any(list(state[2].values()))
        for j in range(max_steps):
            if encoder_type == "lstm":
                if len(obs_seq) == lstm_len:
                    chosen_actions = agent.act(obs_seq)
                    action = dict()
                    for k in range(len(traffic_light_ids)):
                        action[traffic_light_ids[k]] = chosen_actions[k]
                else:
                    action = dict()
                    for k in range(len(traffic_light_ids)):
                        action[traffic_light_ids[k]] = None
            else:
                chosen_actions = agent.act(input_obs)
                action = dict()
                for k in range(len(traffic_light_ids)):
                    action[traffic_light_ids[k]] = chosen_actions[k]
            
            state = env.step(action)
            obs = np.concatenate(list(state[0].values()))
            reward = np.asarray(list(state[1].values())) / float(delta_time)
            reward = np.clip(reward, -1.0, 1.0)
            current_reward_sum += reward
            end = any(list(state[2].values()))
            agent.set_rewards(reward)

            if encoder_type == "lstm":
                input_obs = np.concatenate((obs, reward), axis=0)
                obs_seq.append(input_obs)
                if len(obs_seq) > lstm_len:
                    obs_seq.pop(0)
            else:
                input_obs = np.concatenate((obs, reward, prev_obs, prev_reward), axis=0)
                prev_obs = obs
                prev_reward = reward

            steps += 1
            if end:
                break
            if steps % steps_per_learn == 0:
                if loss_csv is not None:
                    loss = agent.train(return_loss=True)
                    loss_list.append(loss)
                else:
                    agent.train()
                agent.reset_batch()

                current_reward_mean = np.mean(current_reward_sum)
                if current_reward_mean > best_reward_mean:
                    best_reward_mean = current_reward_mean
                    agent.save_model("best_" + model_save_path)

        if reward_csv is not None:
            print(reward_csv.replace("_reward.csv","") + ": episodes " + str(i + 1) + " ended")
        else:
            print("episodes " + str(i + 1) + " ended")

    env.reset()
    env.close()
    agent.reset_batch()
    if loss_csv is not None:
        loss_data = {"loss": loss_list}
        loss_df = pd.DataFrame(loss_data)
        loss_df.to_csv(loss_csv)

    if reward_csv is not None:
        episode_num = list()
        mean_reward = list()
        mean_total_stopped = list()
        mean_total_wait_time = list()
        for i in range(episodes):
            load_path = csv_dir + "_run" + str(i + 1) + ".csv"
            dataframe = pd.read_csv(load_path).dropna(axis=0)
            reward = dataframe["reward"].tolist()
            total_stopped = dataframe["total_stopped"].tolist()
            total_wait_time = dataframe["total_wait_time"].tolist()

            episode_num.append(i + 1)
            mean_reward.append(np.mean(reward))
            mean_total_stopped.append(np.mean(total_stopped))
            mean_total_wait_time.append(np.mean(total_wait_time))

        learn_num = -(-episodes // episode_per_learn)
        episode_num_learn = list()
        mean_reward_learn = list()
        mean_total_stopped_learn = list()
        mean_total_wait_time_learn = list()
        for i in range(learn_num):
            episode_num_learn.append(episode_num[episode_per_learn * i])
            mean_reward_learn.append(np.mean(mean_reward[episode_per_learn * i:episode_per_learn * (i + 1)]))
            mean_total_stopped_learn.append(np.mean(mean_total_stopped[episode_per_learn * i:episode_per_learn * (i + 1)]))
            mean_total_wait_time_learn.append(np.mean(mean_total_wait_time[episode_per_learn * i:episode_per_learn * (i + 1)]))

        analysis_data = {"episode": episode_num_learn, "mean_reward": mean_reward_learn,
                     "mean_total_stopped": mean_total_stopped_learn, "mean_total_wait_time": mean_total_wait_time_learn}
        analysis_dataframe = pd.DataFrame(analysis_data)
        analysis_dataframe.to_csv(reward_csv, index=False)

    agent.save_model(model_save_path)

# 学習させる関数(sumo_rlの実験と同じ設定)
def train_agent_sumorl(
    net_file='/home/kato/traffic_light_control/sumo-rl/nets/2x2grid/2x2.net.xml',
    route_file='/home/kato/traffic_light_control/sumo-rl/nets/2x2grid/2x2.rou.xml',
    model_save_path=None, num_traffic_lights=4, obs_dim=78, num_actions=4, steps_per_learn=100, 
    max_steps=100000, episodes=1, delta_time=1, yellow_time=2, min_green=5, reward_type="waiting_time", 
    num_layers=1, num_hidden_units=128, lr=3e-5, decay_rate=0.01, temperature=1.0, noise=0.0, encoder_type="fc", 
    lstm_len=5, parallel=1, reward_csv=None, loss_csv=None, save_state_dir=None, save_state_interval=50, 
    use_gpu=False):
    agent = Agent(
        num_states=obs_dim*num_traffic_lights, num_traffic_lights=num_traffic_lights, 
        num_actions=num_actions, num_layers=num_layers, num_hidden_units=num_hidden_units,
        temperature=temperature, noise=noise, encoder_type=encoder_type, is_train=True, 
        lr=lr, decay_rate=decay_rate, use_gpu=use_gpu)

    steps = 0
    if reward_csv is not None:
        csv_dir = "./" + reward_csv.replace(".csv","") + "/outputs"
    else:
        csv_dir = None
    
    conn_label = list()
    env_list = list()
    for i in range(parallel):
        if save_state_dir is not None and i==0:
            save_state_dir_i = save_state_dir
        else:
            save_state_dir_i = None
        if reward_csv is not None:
            csv_dir_i = "./" + reward_csv.replace(".csv",str(i+1)) + "/outputs"
        else:
            csv_dir_i = None
        label = "sim" + str(i+1)
        env = SumoEnvironment(
            net_file=net_file,
            route_file=route_file, 
            save_state_dir=save_state_dir_i, save_state_interval=save_state_interval, 
            out_csv_name=csv_dir_i, test=False, use_gui=False, num_seconds=max_steps*10, 
            delta_time=delta_time, yellow_time=yellow_time, min_green=min_green, reward_type=reward_type, 
            label=label, single_agent=False)
        conn_label.append(label)
        env_list.append(env)

    loss_list = list()
    best_reward_mean = float("-inf")
    current_reward = list()
    for i in range(episodes):
        if encoder_type == "lstm":
            obs_seq_list = list()
        else:
            input_obs_list = list()
            prev_obs_list = list()
            prev_reward_list = list()
        end_list = list()
        for j in range(parallel):
            if i != 0:
                traci.switch(conn_label[j])
            reset_obs = env_list[j].reset()
            traffic_light_ids = tuple(reset_obs.keys())
            action = dict()
            for k in range(len(traffic_light_ids)):
                action[traffic_light_ids[k]] = None
            _ = env_list[j].step(action)

            state = env_list[j].step(action)
            obs = np.concatenate(list(state[0].values()))
            reward = np.asarray(list(state[1].values())) / float(delta_time)
            reward = np.clip(reward, -1.0, 1.0)
            if encoder_type == "lstm":
                obs_seq_list.append(list())
                obs_seq_list[j].append(np.concatenate((obs, reward), axis=0))
            else:
                prev_obs = np.zeros(len(obs), dtype=np.float32)
                prev_reward = [0.0 for j in range(len(reward))]
                input_obs_list.append(
                    np.concatenate((obs, reward, prev_obs, prev_reward), axis=0)
                )
                prev_obs_list.append(obs)
                prev_reward_list.append(reward)
            end_list.append(any(list(state[2].values())))

        for j in range(max_steps):
            for k in range(parallel):
                if end_list[k]:
                    continue
                traci.switch(conn_label[k])

                if encoder_type == "lstm":
                    if len(obs_seq_list[k]) == lstm_len:
                        chosen_actions = agent.act(obs_seq_list[k])
                        action = dict()
                        for m in range(len(traffic_light_ids)):
                            action[traffic_light_ids[m]] = chosen_actions[m]
                    else:
                        action = dict()
                        for m in range(len(traffic_light_ids)):
                            action[traffic_light_ids[m]] = None
                else:
                    chosen_actions = agent.act(input_obs_list[k])
                    action = dict()
                    for m in range(len(traffic_light_ids)):
                        action[traffic_light_ids[m]] = chosen_actions[m]
            
                state = env_list[k].step(action)
                
                obs = np.concatenate(list(state[0].values()))
                reward = np.asarray(list(state[1].values())) / float(delta_time)
                reward = np.clip(reward, -1.0, 1.0)
                current_reward.append(reward)
                end_list[k] = any(list(state[2].values()))
                agent.set_rewards(reward)
                
                if encoder_type == "lstm":
                    obs_seq_list[k].append(np.concatenate((obs, reward), axis=0))
                    if len(obs_seq_list[k]) > lstm_len:
                        obs_seq_list[k].pop(0)
                else:
                    input_obs_list[k] = np.concatenate(
                        (obs, reward, prev_obs, prev_reward), axis=0)
                    prev_obs_list[k] = obs
                    prev_reward_list[k] = reward

            steps += 1
            if steps % steps_per_learn == 0:
                if loss_csv is not None:
                    loss = agent.train(return_loss=True)
                    loss_list.append(loss)
                else:
                    agent.train()
                agent.reset_batch()

                current_reward_mean = np.mean(current_reward)
                if current_reward_mean > best_reward_mean:
                    best_reward_mean = current_reward_mean
                    agent.save_model("best_" + model_save_path)
                if reward_csv is not None:
                    print(reward_csv.replace("_reward.csv", "") + ": steps " + str(j + 1) + " ended")
                else:
                    print("steps " + str(j + 1) + " ended")

        if reward_csv is not None:
            print(reward_csv.replace("_reward.csv","") + ": episodes " + str(i + 1) + " ended")
        else:
            print("episodes " + str(i + 1) + " ended")

    for i in range(parallel):
        traci.switch(conn_label[i])
        env_list[i].reset()
        env_list[i].close()
    agent.reset_batch()
    if loss_csv is not None:
        loss_data = {"loss": loss_list}
        loss_df = pd.DataFrame(loss_data)
        loss_df.to_csv(loss_csv)

    if reward_csv is not None:
        for i in range(episodes):
            step_num = list()
            mean_reward = list()
            mean_total_stopped = list()
            mean_total_wait_time = list()

            reward_temp = list()
            total_stopped_temp = list()
            total_wait_time_temp = list()
            max_length = 0
            for j in range(parallel):
                load_path = "./" + reward_csv.replace(".csv",str(j+1)) + "/outputs_run" + str(i + 1) + ".csv"
                dataframe = pd.read_csv(load_path).dropna(axis=0)
                reward_temp.append(dataframe["reward"].tolist())
                total_stopped_temp.append(dataframe["total_stopped"].tolist())
                total_wait_time_temp.append(dataframe["total_wait_time"].tolist())
                if len(reward_temp[j]) > max_length:
                    max_length = len(reward_temp[j])
            
            reward = np.zeros(max_length)
            total_stopped = np.zeros(max_length)
            total_wait_time = np.zeros(max_length)
            for j in range(parallel):
                reward += np.pad(reward_temp[j], ((0,max_length-len(reward_temp[j])),))
                total_stopped += np.pad(total_stopped_temp[j], ((0,max_length-len(total_stopped_temp[j])),))
                total_wait_time += np.pad(total_wait_time_temp[j], ((0,max_length-len(total_wait_time_temp[j])),))
            reward /= parallel
            total_stopped /= parallel
            total_wait_time /= parallel

            learn_num = -(-max_length // steps_per_learn)
            for j in range(learn_num):
                step_num.append(steps_per_learn*j + 1)
                mean_reward.append(np.mean(reward[steps_per_learn * j:steps_per_learn * (j + 1)]))
                mean_total_stopped.append(np.mean(total_stopped[steps_per_learn * j:steps_per_learn * (j + 1)]))
                mean_total_wait_time.append(np.mean(total_wait_time[steps_per_learn * j:steps_per_learn * (j + 1)]))

            analysis_data = {"step": step_num, "mean_reward": mean_reward,
                     "mean_total_stopped": mean_total_stopped, "mean_total_wait_time": mean_total_wait_time}
            analysis_dataframe = pd.DataFrame(analysis_data)
            analysis_path = csv_dir.replace("outputs", "episode"+str(i+1)) + ".csv"
            analysis_dataframe.to_csv(analysis_path, index=False)

    agent.save_model(model_save_path)