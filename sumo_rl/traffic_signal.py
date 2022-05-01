import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import numpy as np
from gym import spaces


class TrafficSignal:
    """
    This class represents a Traffic Signal of an intersection
    It is responsible for retrieving information and changing the traffic phase using Traci API
    """

    def __init__(self, env, ts_id, delta_time, yellow_time, min_green):
        self.id = ts_id
        self.env = env
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.green_phase = 0
        self.is_yellow = False
        self.time_since_last_phase_change = 0
        self.next_action_time = 0
        self.last_measure = 0.0
        self.last_reward = None

        self.build_phases()

        self.lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(self.id)))  # Remove duplicates and keep order
        self.out_lanes = [link[0][1] for link in traci.trafficlight.getControlledLinks(self.id) if link]
        self.out_lanes = list(set(self.out_lanes))
        self.lanes_length = {lane: traci.lane.getLength(lane) for lane in self.lanes}

        self.observation_space = spaces.Box(low=np.zeros(self.num_green_phases+1+2*len(self.lanes), dtype=np.float32), high=np.ones(self.num_green_phases+1+2*len(self.lanes), dtype=np.float32))
        self.discrete_observation_space = spaces.Tuple((
            spaces.Discrete(self.num_green_phases),                       # Green Phase
            spaces.Discrete(2),                                           # Binary variable active if min_green seconds already elapsed
            *(spaces.Discrete(10) for _ in range(2*len(self.lanes)))      # Density and stopped-density for each lane
        ))
        self.action_space = spaces.Discrete(self.num_green_phases)


    def build_phases(self):
        phases = traci.trafficlight.getAllProgramLogics(self.id)[0].phases

        self.green_phases = list()
        self.yellow_dict = dict()
        for phase in phases:
            state = phase.state
            if 'y' not in state and (state.count('r') + state.count('s') != len(state)):
                self.green_phases.append(traci.trafficlight.Phase(60, state))
        self.num_green_phases = len(self.green_phases)
        self.all_phases = self.green_phases.copy()

        for i, p1 in enumerate(self.green_phases):
            for j, p2 in enumerate(self.green_phases):
                if i == j:
                    continue
                yellow_state = ''
                for s in range(len(p1.state)):
                    if (p1.state[s] == 'G' or p1.state[s] == 'g') and (p2.state[s] == 'r' or p2.state[s] == 's'):
                        yellow_state += 'y'
                    else:
                        yellow_state += p1.state[s]
                self.yellow_dict[(i,j)] = len(self.all_phases)
                self.all_phases.append(traci.trafficlight.Phase(self.yellow_time, yellow_state))

        programs = traci.trafficlight.getAllProgramLogics(self.id)
        logic = programs[0]
        logic.type = 0
        logic.phases = self.all_phases
        traci.trafficlight.setProgramLogic(self.id, logic)
        traci.trafficlight.setRedYellowGreenState(self.id, self.all_phases[0].state)

    @property
    def time_to_act(self):
        return self.next_action_time == self.env.sim_step
    
    def update(self):
        self.time_since_last_phase_change += 1
        if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
            #traci.trafficlight.setPhase(self.id, self.green_phase)
            traci.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.is_yellow = False

    def set_next_phase(self, new_phase):
        """
        Sets what will be the next green phase and sets yellow phase if the next phase is different than the current
        :param new_phase: (int) Number between [0..num_green_phases] 
        """
        if new_phase is not None:
            new_phase = int(new_phase)
        
        if new_phase is None or self.green_phase == new_phase or self.time_since_last_phase_change < self.yellow_time + self.min_green:
            #traci.trafficlight.setPhase(self.id, self.green_phase)
            traci.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            if self.time_since_last_phase_change < self.yellow_time + self.min_green:
                self.next_action_time = max(
                    [self.env.sim_step + self.min_green + self.yellow_time - self.time_since_last_phase_change, 
                    self.env.sim_step + self.delta_time]
                    )
            else:
                self.next_action_time = self.env.sim_step + self.delta_time
        else:
            #traci.trafficlight.setPhase(self.id, self.yellow_dict[(self.green_phase, new_phase)])  # turns yellow
            traci.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.yellow_dict[(self.green_phase, new_phase)]].state)
            self.green_phase = new_phase
            self.next_action_time = self.env.sim_step + self.delta_time
            self.is_yellow = True
            self.time_since_last_phase_change = 0
    
    def compute_observation(self):
        time_info = self.compute_time_for_observation()
        phase_id = [1 if self.phase//2 == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        density = self.get_lanes_density()
        queue = self.get_lanes_queue()
        distance, speed = self.get_distance_and_speed()
        observation = np.array(time_info + phase_id + density + queue + distance + speed, dtype=np.float32)
        return observation
            
    def compute_reward(self):
        if self.env.reward_type == "waiting_time":
            self.last_reward = self._waiting_time_reward()
        elif self.env.reward_type == "vehicle_speed":
            self.last_reward = self._vehicle_speed_reward()
        elif self.env.reward_type == "vehicle_distance":
            self.last_reward = self._vehicle_distance_reward()
        return self.last_reward
    
    def _pressure_reward(self):
        return -self.get_pressure()

    def _queue_average_reward(self):
        new_average = np.mean(self.get_stopped_vehicles_num())
        reward = self.last_measure - new_average
        self.last_measure = new_average
        return reward

    def _queue_reward(self):
        return - (sum(self.get_stopped_vehicles_num()))**2

    def _waiting_time_reward(self):
        ts_wait = sum(self.get_waiting_time_per_lane()) / 100.0
        reward = self.last_measure - ts_wait
        self.last_measure = ts_wait
        return reward

    def _waiting_time_reward2(self):
        ts_wait = sum(self.get_waiting_time())
        self.last_measure = ts_wait
        if ts_wait == 0:
            reward = 1.0
        else:
            reward = 1.0/ts_wait
        return reward

    def _waiting_time_reward3(self):
        ts_wait = sum(self.get_waiting_time())
        reward = -ts_wait
        self.last_measure = ts_wait
        return reward

    def _vehicle_speed_reward(self):
        veh_speed = list()
        for lane in self.lanes:
            veh_list = traci.lane.getLastStepVehicleIDs(lane)
            for veh in veh_list:
                speed = traci.vehicle.getSpeed(veh)
                speed_norm = speed / 10.0
                veh_speed.append(speed_norm)

        if len(veh_speed) == 0:
            veh_speed_mean = 0.0
        else:
            veh_speed_mean = np.mean(veh_speed).tolist()
        return veh_speed_mean

    def _vehicle_distance_reward(self):
        veh_dist = list()
        for lane in self.lanes:
            veh_list = traci.lane.getLastStepVehicleIDs(lane)
            for veh in veh_list:
                leader = traci.vehicle.getLeader(veh)
                if leader is None:
                    continue
                else:
                    dist_norm = leader[1] / 10.0
                    veh_dist.append(dist_norm)

        if len(veh_dist) == 0:
            veh_dist_mean = 0.0
        else:
            veh_dist_mean = np.mean(veh_dist).tolist()
        return veh_dist_mean


    def get_waiting_time_per_lane(self):
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list = traci.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = traci.vehicle.getLaneID(veh)
                acc = traci.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum([self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane])
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane

    def get_pressure(self):
        return abs(sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes) - sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes))

    def get_out_lanes_density(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, traci.lane.getLastStepVehicleNumber(lane) / (traci.lane.getLength(lane) / vehicle_size_min_gap)) for lane in self.out_lanes]

    def get_lanes_density(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, traci.lane.getLastStepVehicleNumber(lane) / (traci.lane.getLength(lane) / vehicle_size_min_gap)) for lane in self.lanes]

    def get_lanes_queue(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, traci.lane.getLastStepHaltingNumber(lane) / (traci.lane.getLength(lane) / vehicle_size_min_gap)) for lane in self.lanes]
    
    def get_total_queued(self):
        return sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.lanes])

    def _get_veh_list(self):
        veh_list = []
        for lane in self.lanes:
            veh_list += traci.lane.getLastStepVehicleIDs(lane)
        return veh_list

    def get_distance_and_speed(self):
        veh_dist_mean = list()
        veh_speed_mean = list()
        for lane in self.lanes:
            veh_dist = list()
            veh_speed = list()
            veh_list = traci.lane.getLastStepVehicleIDs(lane)
            for veh in veh_list:
                speed = traci.vehicle.getSpeed(veh)
                max_speed = traci.vehicle.getMaxSpeed(veh)
                speed_norm = speed / max_speed
                veh_speed.append(speed_norm)

                leader = traci.vehicle.getLeader(veh)
                if leader is None:
                    continue
                else:
                    standard_len = traci.lane.getLength(lane)
                    dist_norm = leader[1] / standard_len
                    if abs(dist_norm) > 1.0:
                        dist_norm = 1.0
                    veh_dist.append(dist_norm)

            if len(veh_dist) == 0:
                veh_dist_mean.append(1.0)
            else:
                veh_dist_mean.append(np.mean(veh_dist).tolist())

            if len(veh_speed) == 0:
                veh_speed_mean.append(1.0)
            else:
                veh_speed_mean.append(np.mean(veh_speed).tolist())

        return veh_dist_mean, veh_speed_mean

    def compute_time_for_observation(self):
        time_norm = self.time_since_last_phase_change/(self.yellow_time + self.min_green*3)
        if time_norm>1.0:
            time_norm = 1.0
        return [float(self.time_to_act), time_norm]