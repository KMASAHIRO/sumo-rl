from lib2to3.pytree import LeafPattern
import os
import sys
from pathlib import Path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import sumolib
import gym
from gym.envs.registration import EnvSpec
import numpy as np
import pandas as pd

from .traffic_signal import TrafficSignal


class SumoEnvironment(gym.Env):
    """
    SUMO Environment for Traffic Signal Control

    :param net_file: (str) SUMO .net.xml file
    :param route_file: (str) SUMO .rou.xml file
    :param phases: (traci.trafficlight.Phase list) Traffic Signal phases definition
    :param out_csv_name: (str) name of the .csv output with simulation results. If None no output is generated
    :param use_gui: (bool) Wheter to run SUMO simulation with GUI visualisation
    :param num_seconds: (int) Number of simulated seconds on SUMO
    :param max_depart_delay: (int) Vehicles are discarded if they could not be inserted after max_depart_delay seconds
    :param delta_time: (int) Simulation seconds between actions
    :param min_green: (int) Minimum green time in a phase
    :single_agent: (bool) If true, it behaves like a regular gym.Env. Else, it behaves like a MultiagentEnv (https://github.com/ray-project/ray/blob/master/python/ray/rllib/env/multi_agent_env.py)
    """

    def __init__(
        self, net_file, route_file, save_state_dir=None, save_state_interval=1, out_csv_name=None, test=False, 
        use_gui=False, begin_seconds=0, num_seconds=20000.0, max_depart_delay=100000, time_to_teleport=-1, delta_time=5, yellow_time=2, 
        min_green=5, reward_type="waiting_time", port=None, label="sim1", single_agent=False, seed="random"):
        self._net = net_file
        self._route = route_file
        self.use_gui = use_gui
        if self.use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        self.begin_seconds = begin_seconds
        self.sim_max_time = begin_seconds + num_seconds
        self.delta_time = delta_time  # seconds on sumo at each step
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.time_to_teleport = time_to_teleport
        self.min_green = min_green
        self.yellow_time = yellow_time
        self.reward_type = reward_type
        self.port = port
        self.label = label
        self.seed = seed

        
        self.single_agent = single_agent
        self.vehicles = dict()

        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {}
        self.spec = EnvSpec('SUMORL-v0')

        self.run = 0
        self.metrics = []
        self.out_csv_name = out_csv_name
        
        self.smooth_record = test
        self.step_num = 0
        self.save_state_interval = save_state_interval
        if save_state_dir is None:
            self.save_state_dir = None
            self.save_state_flag = False
        else:
            if not os.path.exists(save_state_dir):
                raise FileExistsError("directory does not exist")
            self.save_state_dir = save_state_dir
            self.save_state_flag = True
    
    def reset(self):
        if self.run != 0:
            traci.close()
            self.save_csv(self.out_csv_name, self.run)
        self.run += 1
        self.metrics = []
        self.step_num = 0
        if self.save_state_dir is not None:
            os.mkdir(os.path.join(self.save_state_dir, "sumo_state_run"+str(self.run)+"/"))

        sumo_cmd = [self._sumo_binary,
                     '-n', self._net,
                     '-r', self._route,
                     '--max-depart-delay', str(self.max_depart_delay), 
                     '--waiting-time-memory', '10000',
                     '--time-to-teleport', str(self.time_to_teleport),
                     '-b', str(self.begin_seconds)]

        if self.seed == "random":
            sumo_cmd.append("--random")
        else:
            sumo_cmd.extend(["--seed", str(self.seed)])

        if self.use_gui:
            sumo_cmd.append('--start')

        traci.start(sumo_cmd, port=self.port, label=self.label)

        self.ts_ids = list(traci.trafficlight.getIDList())

        self.traffic_signals = {ts: TrafficSignal(self, ts, self.delta_time, self.yellow_time, self.min_green, self.begin_seconds) for ts in self.ts_ids}

        self.vehicles = dict()

        self.save_state("step0.xml", self.run)
        
        if self.single_agent:
            return self._compute_observations()[self.ts_ids[0]]
        else:
            return self._compute_observations()

    @property
    def sim_step(self):
        """
        Return current simulation second on SUMO
        """
        return traci.simulation.getTime()

    def step(self, action):
        # No action, follow fixed TL defined in self.phases
        if action is None or action == {}:
            self._apply_actions(None)
        else:
            self._apply_actions(action)

        for i in range(self.delta_time):
            self._sumo_step()

            if self.smooth_record:
                self.step_num += 1
                if self.step_num % self.save_state_interval == 0:
                    save_file = "step" + str(self.step_num) + ".xml"
                    self.save_state(save_file, self.run)

            for ts in self.ts_ids:
                self.traffic_signals[ts].update()

            if self.sim_step % self.delta_time == 0:
                info = self._compute_step_info()
                self.metrics.append(info)

        observations = self._compute_observations()
        rewards = self._compute_rewards()
        done = {'__all__': self.sim_step > self.sim_max_time}
        done.update({ts_id: False for ts_id in self.ts_ids})

        if not self.smooth_record:
            self.step_num += 1
            if self.step_num % self.save_state_interval == 0:
                save_file = "step" + str(self.step_num) + ".xml"
                self.save_state(save_file, self.run)

        if self.single_agent:
            return observations[self.ts_ids[0]], rewards[self.ts_ids[0]], done['__all__'], {}
        else:
            return observations, rewards, done, {}

    """
    def step(self, action):
        # No action, follow fixed TL defined in self.phases
        if action is None or action == {}:
            self._apply_actions(None)
        else:
            self._apply_actions(action)

        time_to_act = False
        while not time_to_act:
            self._sumo_step()

            if self.smooth_record:
                self.step_num += 1
                save_file = "step" + str(self.step_num) + ".xml"
                self.save_state(save_file, self.run)

            for ts in self.ts_ids:
                self.traffic_signals[ts].update()
                if self.traffic_signals[ts].time_to_act:
                    time_to_act = True

            if self.sim_step % 5 == 0:
                info = self._compute_step_info()
                self.metrics.append(info)

        observations = self._compute_observations()
        rewards = self._compute_rewards()
        done = {'__all__': self.sim_step > self.sim_max_time}
        done.update({ts_id: False for ts_id in self.ts_ids})
        
        if not self.smooth_record:
            self.step_num += 1
            save_file = "step" + str(self.step_num) + ".xml"
            self.save_state(save_file, self.run)

        if self.single_agent:
            return observations[self.ts_ids[0]], rewards[self.ts_ids[0]], done['__all__'], {}
        else:
            return observations, rewards, done, {}
    """

    def _apply_actions(self, actions):
        """
        Set the next green phase for the traffic signals
        :param actions: If single-agent, actions is an int between 0 and self.num_green_phases (next green phase)
                        If multiagent, actions is a dict {ts_id : greenPhase}
        """   
        if self.single_agent:
            self.traffic_signals[self.ts_ids[0]].set_next_phase(actions)
        else:
            for ts, action in actions.items():
                self.traffic_signals[ts].set_next_phase(action)
    
    def _compute_observations(self):
        #return {ts: self.traffic_signals[ts].compute_observation() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act}
        return {ts: self.traffic_signals[ts].compute_observation() for ts in self.ts_ids}

    def _compute_rewards(self):
        #return {ts: self.traffic_signals[ts].compute_reward() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act}
        return {ts: self.traffic_signals[ts].compute_reward() for ts in self.ts_ids}

    @property
    def observation_space(self):
        return self.traffic_signals[self.ts_ids[0]].observation_space
    
    @property
    def action_space(self):
        return self.traffic_signals[self.ts_ids[0]].action_space
    
    def observation_spaces(self, ts_id):
        return self.traffic_signals[ts_id].observation_space
    
    def action_spaces(self, ts_id):
        return self.traffic_signals[ts_id].action_space

    def _sumo_step(self):
        traci.simulationStep()

    def _compute_step_info(self):
        return {
            'step_time': self.sim_step,
            'reward': sum(0 if self.traffic_signals[ts].last_reward is None else self.traffic_signals[ts].last_reward for ts in self.ts_ids),
            'total_stopped': sum(self.traffic_signals[ts].get_total_queued() for ts in self.ts_ids),
            'total_wait_time': sum(sum(self.traffic_signals[ts].get_waiting_time_per_lane()) for ts in self.ts_ids)
        }

    def close(self):
        traci.close()
    
    def save_csv(self, out_csv_name, run):
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            Path(Path(out_csv_name).parent).mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv_name + '_run{}'.format(run) + '.csv', index=False)

    # Below functions are for discrete state space
    
    def save_state(self, file_name, run):
        if self.save_state_dir is not None and self.save_state_flag:
            path = os.path.join(self.save_state_dir, "sumo_state_run"+str(run)+"/", file_name)
            traci.simulation.saveState(path)

    def encode(self, state, ts_id):
        phase = int(np.where(state[:self.traffic_signals[ts_id].num_green_phases] == 1)[0])
        #elapsed = self._discretize_elapsed_time(state[self.num_green_phases])
        density_queue = [self._discretize_density(d) for d in state[self.traffic_signals[ts_id].num_green_phases:]]
        # tuples are hashable and can be used as key in python dictionary
        return tuple([phase] + density_queue)

    def _discretize_density(self, density):
        return min(int(density*10), 9)


    
