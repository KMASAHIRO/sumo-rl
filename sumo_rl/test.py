import os
import torch
import numpy as np
import pandas as pd
import traci

from .env import SumoEnvironment
from .module import Agent

# 学習させる関数
def test_agent(
    net_file='/home/kato/traffic_light_control/sumo-rl/nets/2x2grid/2x2.net.xml',
    route_file='/home/kato/traffic_light_control/sumo-rl/nets/2x2grid/2x2.rou.xml',
    model_load_path=None, num_traffic_lights=4, obs_dim=78, num_actions=4, max_steps=300, 
    episodes=10000, delta_time=1, yellow_time=2, min_green=5, reward_type="waiting_time", 
    num_layers=1, num_hidden_units=512, encoder_type="fc", lstm_len=5, embedding_num=5, 
    reward_csv=None, use_gpu=False, use_gui=False, seed=0):
    agent = Agent(
        num_states=obs_dim*num_traffic_lights, num_traffic_lights=num_traffic_lights, 
        num_actions=num_actions, num_layers=num_layers, num_hidden_units=num_hidden_units, 
        temperature=1.0, noise=0.0, encoder_type=encoder_type, lr=3e-5, decay_rate=0.01, 
        embedding_num=embedding_num, embedding_decay=0.99, eps=1e-5, beta=0.25, is_train=False, 
        use_gpu=use_gpu, model_path=model_load_path)
    
    if reward_csv is not None:
        csv_dir = "./" + reward_csv.replace(".csv","") + "/outputs"
    else:
        csv_dir = None
    env = SumoEnvironment(net_file=net_file,
                          route_file=route_file,
                          save_state_dir=None,
                          out_csv_name=csv_dir, test=False, use_gui=use_gui,
                          delta_time=delta_time, yellow_time=yellow_time, 
                          min_green=min_green, reward_type=reward_type, 
                          single_agent=False)
    
    if model_load_path == "random":
        np.random.seed(seed)
    for i in range(episodes):
        if model_load_path != "random":
            if encoder_type == "lstm":
                obs_seq = list()
        reset_obs = env.reset()
        traffic_light_ids = tuple(reset_obs.keys())
        action = dict()
        for j in range(len(traffic_light_ids)):
            action[traffic_light_ids[j]] = None
        _ = env.step(action)

        state = env.step(action)
        if model_load_path != "random":
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
            if model_load_path == "random":
                action = dict()
                action_random = np.random.randint(0,4,len(traffic_light_ids))
                for k in range(len(traffic_light_ids)):
                    action[traffic_light_ids[k]] = action_random[k]
            else:
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
            if model_load_path != "random":
                obs = np.concatenate(list(state[0].values()))
                reward = np.asarray(list(state[1].values())) / float(delta_time)
                reward = np.clip(reward, -1.0, 1.0)
                if encoder_type == "lstm":
                    input_obs = np.concatenate((obs, reward), axis=0)
                    obs_seq.append(input_obs)
                    if len(obs_seq) > lstm_len:
                        obs_seq.pop(0)
                else:
                    input_obs = np.concatenate((obs, reward, prev_obs, prev_reward), axis=0)
                    prev_obs = obs
                    prev_reward = reward
            end = any(list(state[2].values()))

            if end:
                break
            

        if reward_csv is not None:
            print(reward_csv.replace("_reward.csv","") + ": episodes " + str(i + 1) + " ended")
        else:
            print("episodes " + str(i + 1) + " ended")

    env.reset()
    env.close()

    if reward_csv is not None:
        whole_reward = list()
        whole_total_stopped = list()
        whole_total_wait_time = list()
        for i in range(episodes):
            load_path = csv_dir + "_run" + str(i + 1) + ".csv"
            dataframe = pd.read_csv(load_path).dropna(axis=0)
            reward = dataframe["reward"].tolist()
            total_stopped = dataframe["total_stopped"].tolist()
            total_wait_time = dataframe["total_wait_time"].tolist()

            whole_reward.append(reward)
            whole_total_stopped.append(total_stopped)
            whole_total_wait_time.append(total_wait_time)
        
        mean_reward = np.mean(whole_reward, axis=1)
        mean_total_stopped = np.mean(whole_total_stopped, axis=1)
        mean_total_wait_time = np.mean(whole_total_wait_time, axis=1)

        analysis_data = {"mean_reward": mean_reward,
                     "mean_total_stopped": mean_total_stopped, 
                     "mean_total_wait_time": mean_total_wait_time}
        analysis_dataframe = pd.DataFrame(analysis_data)
        analysis_dataframe.to_csv(reward_csv, index=False)

# 学習させる関数(sumo_rlの実験と同じ設定)
def test_agent_sumorl(
    net_file='/home/kato/traffic_light_control/sumo-rl/nets/2x2grid/2x2.net.xml',
    route_file='/home/kato/traffic_light_control/sumo-rl/nets/2x2grid/2x2.rou.xml',
    model_load_path=None, num_traffic_lights=4, obs_dim=78, num_actions=4, steps_per_learn=100, 
    max_steps=100000, episodes=1, delta_time=1, yellow_time=2, min_green=5, reward_type="waiting_time", 
    num_layers=1, num_hidden_units=128, encoder_type="fc", lstm_len=5, parallel=1, reward_csv=None, 
    save_state_dir=None, save_state_interval=50, use_gpu=False, use_gui=False, seed=0):
    agent = Agent(
        num_states=obs_dim*num_traffic_lights, num_traffic_lights=num_traffic_lights, 
        num_actions=num_actions, num_layers=num_layers, num_hidden_units=num_hidden_units,
        temperature=1.0, encoder_type=encoder_type, is_train=False, use_gpu=use_gpu, 
        model_path=model_load_path)

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
    
    if model_load_path == "random":
        np.random.seed(seed)
    for i in range(episodes):
        steps = 0
        if model_load_path != "random":
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
            if model_load_path != "random":
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

                if model_load_path == "random":
                    action_random = np.random.randint(0,4,len(traffic_light_ids))
                    action = dict()
                    for m in range(len(traffic_light_ids)):
                        action[traffic_light_ids[m]] = action_random[m]
                else:
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
                
                if model_load_path != "random":
                    obs = np.concatenate(list(state[0].values()))
                    reward = np.asarray(list(state[1].values())) / float(delta_time)
                    reward = np.clip(reward, -1.0, 1.0)
                
                    if encoder_type == "lstm":
                        obs_seq_list[k].append(np.concatenate((obs, reward), axis=0))
                        if len(obs_seq_list[k]) > lstm_len:
                            obs_seq_list[k].pop(0)
                    else:
                        input_obs_list[k] = np.concatenate(
                            (obs, reward, prev_obs, prev_reward), axis=0)
                        prev_obs_list[k] = obs
                        prev_reward_list[k] = reward
                end_list[k] = any(list(state[2].values()))

            steps += 1
            if steps % steps_per_learn == 0:
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