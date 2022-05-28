import torch
import numpy as np

# 方策関数
class PolicyFunction(torch.nn.Module):
    def __init__(
        self, num_states, num_traffic_lights, num_actions, num_layers=1, 
        num_hidden_units=128, temperature=1.0, noise=0.0, encoder_type="fc", 
        embedding_type="random", embedding_num=5, embedding_decay=0.99, eps=1e-5, 
        device="cpu"):
        
        super().__init__()
        self.num_states = num_states
        self.num_traffic_lights = num_traffic_lights
        self.num_actions = num_actions
        self.temperature = temperature
        self.noise = noise
        self.encoder_type = encoder_type
        self.embedding_num = embedding_num
        self.embedding_decay = embedding_decay
        self.eps = eps
        self.device = torch.device(device)

        if self.encoder_type == "fc":
            self.fc_first = torch.nn.Linear(self.num_states, num_hidden_units)
            self.encoder = self.fc_encoder
        elif self.encoder_type == "lstm":
            self.lstm = torch.nn.LSTM(self.num_states, num_hidden_units, 1, batch_first=True)
            self.fc_first = torch.nn.Linear(num_hidden_units, num_hidden_units)
            self.encoder = self.lstm_encoder
        elif self.encoder_type == "vq":
            self.fc_first = torch.nn.Linear(self.num_states, num_hidden_units)
            if embedding_type == "random":
                embedding = torch.randn(self.embedding_num, num_hidden_units)
            elif embedding_type == "one_hot":
                embedding = torch.nn.functional.one_hot(torch.tensor(range(num_hidden_units)), num_classes=num_hidden_units)
            self.embedding = torch.nn.Parameter(embedding, requires_grad=False)
            self.embedding_avg = torch.nn.Parameter(embedding, requires_grad=False)
            self.cluster_size = torch.nn.Parameter(torch.zeros(self.embedding_num), requires_grad=False)
            self.encoder = self.vq_encoder
        
        
        fc_last_layers = list()
        for i in range(num_traffic_lights):
            fc_last_layers.append(torch.nn.Linear(num_hidden_units, self.num_actions[i]))
        self.fc_last_layers = torch.nn.ModuleList(fc_last_layers)

        fc_layers = list()
        self.num_layers = num_layers
        for i in range(num_layers):
            fc_layers.append(torch.nn.Linear(num_hidden_units, num_hidden_units))
        self.fc_layers = torch.nn.ModuleList(fc_layers)

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)
    
    def fc_encoder(self, inputs):
        x = self.fc_first(inputs)
        x = self.relu(x)
        for i in range(self.num_layers):
            x = self.fc_layers[i](x)
            x = self.relu(x)
        
        if self.training:
            x = x + torch.normal(torch.zeros(x.shape[-1]), torch.ones(x.shape[-1])*self.noise).to(self.device)
        
        return x
    
    def lstm_encoder(self, inputs):
        x = inputs.unsqueeze(0)
        _, (x, _) = self.lstm(x)
        x = x.reshape(-1)
        x = self.fc_first(x)

        x = self.relu(x)
        for i in range(self.num_layers):
            x = self.fc_layers[i](x)
            x = self.relu(x)
        
        if self.training:
            x = x + torch.normal(torch.zeros(x.shape[-1]), torch.ones(x.shape[-1])*self.noise).to(self.device)
        
        return x
    
    def vq_encoder(self, inputs):
        x = self.fc_first(inputs)
        x = self.relu(x)
        for i in range(self.num_layers):
            x = self.fc_layers[i](x)
            x = self.relu(x)
        
        vector = x.detach()
        embedding_idx = (vector - self.embedding).pow(2).sum(-1).argmin(-1)

        quantize = x + (self.embedding[embedding_idx] - vector)
        if self.training:
            beta_loss = (x - self.embedding[embedding_idx]).pow(2).mean(-1)
            return quantize, beta_loss, vector, embedding_idx
        else:
            return quantize

    def forward(self, inputs):
        if self.encoder_type == "vq" and self.training:
            x, beta_loss, vector, embedding_idx = self.encoder(inputs)
        else:
            x = self.encoder(inputs)

        last_outputs = list()
        for i in range(self.num_traffic_lights):
            last_x = self.fc_last_layers[i](x)
            last_x = self.softmax(last_x/self.temperature)
            last_outputs.append(last_x)
        
        if self.encoder_type == "vq" and self.training:
            return last_outputs, beta_loss, vector, embedding_idx
        else:
            return last_outputs

# 損失関数(期待収益のマイナス符号)
class PolicyGradientLossWithREINFORCE(torch.nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, actions_prob_history, rewards_history, beta=0.25, beta_loss_history=None):
        ave_rewards = np.mean(rewards_history)
        loss = 0
        for i in range(len(actions_prob_history)):
            chosen_action_prob = actions_prob_history[i]
            # 最大化する関数のマイナス
            loss = loss - torch.log(chosen_action_prob) * (rewards_history[i] - ave_rewards)
        
        loss = loss / len(actions_prob_history)
        
        if beta_loss_history is not None:
            vq_loss = 0
            for i in range(len(beta_loss_history)):
                # 埋め込み前のベクトルが埋め込み表現から一気にずれ過ぎないようにする損失
                vq_loss = vq_loss + beta*beta_loss_history[i]
            vq_loss = vq_loss / len(beta_loss_history)
            loss = loss + vq_loss

        return loss


# エージェント(方策関数を学習し、行動を決定する)
class Agent():
    def __init__(
        self, num_states, num_traffic_lights, num_actions, num_layers, num_hidden_units, 
        temperature, noise, encoder_type, lr, decay_rate, embedding_num, embedding_decay, 
        eps, beta, embedding_no_train=False, embedding_start_train=None, is_train=True, 
        device="cpu", model_path=None
        ):
        
        self.num_states = num_states
        self.num_traffic_lights = num_traffic_lights
        self.num_actions = num_actions
        self.lr = lr
        self.encoder_type = encoder_type
        self.beta = beta
        self.embedding_no_train = embedding_no_train
        self.embedding_start_train = embedding_start_train
        self.is_train = is_train
        self.train_num = 0
        self.device = torch.device(device)
        
        self.policy_function = PolicyFunction(
            self.num_states, self.num_traffic_lights, self.num_actions, num_layers, 
            num_hidden_units, temperature, noise, encoder_type, embedding_num, 
            embedding_decay, eps, device)
        self.policy_function.to(self.device)

        if model_path:
            self.policy_function.load_state_dict(torch.load(model_path))

        if self.is_train:
            self.actions_prob_history = list()
            self.rewards_history = list()
            if self.encoder_type == "vq":
                self.middle_outputs = list()
                for i in range(embedding_num):
                    self.middle_outputs.append(list())
                self.beta_loss_history = list()
            self.policy_function.train()
            self.loss_f = PolicyGradientLossWithREINFORCE()
            param_optimizer = list(self.policy_function.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': decay_rate},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            self.optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=self.lr)
        else:
            self.policy_function.eval()

    def act(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)

        if self.is_train:
            if self.encoder_type == "vq":
                actions_prob, beta_loss, vector, embedding_idx = self.policy_function(obs)
            else:
                actions_prob = self.policy_function(obs)
        else:
            with torch.no_grad():
                actions_prob = self.policy_function(obs)
        
        chosen_actions = list()
        for i in range(self.num_traffic_lights):
            prob_numpy = actions_prob[i].to("cpu").detach().numpy()
            chosen_actions.append(
                np.random.choice(
                    a=np.arange(len(prob_numpy)), 
                    size=1, 
                    replace=True, 
                    p=prob_numpy
                    )[0]
                )

        if self.is_train:
            for i in range(self.num_traffic_lights):
                self.actions_prob_history.append(actions_prob[i][chosen_actions[i]])
            if self.encoder_type == "vq":
                self.middle_outputs[embedding_idx].append(vector)
                self.beta_loss_history.append(beta_loss)
        
        if self.device != "cpu":
            torch.cuda.empty_cache()
        
        return chosen_actions

    def train(self, return_loss=False):
        self.train_num += 1
        if self.embedding_start_train is not None:
            if self.train_num == self.embedding_start_train:
                self.embedding_no_train = False
        
        if self.encoder_type == "vq":
            if self.embedding_no_train:
                self.middle_outputs = list()
                for i in range(self.policy_function.embedding_num):
                    self.middle_outputs.append(list())
                loss = self.loss_f(self.actions_prob_history, self.rewards_history, self.beta, self.beta_loss_history)
            else:
                prev_embedding_avg = self.policy_function.embedding_avg.to("cpu")
                prev_cluster_size = self.policy_function.cluster_size.to("cpu")
                decay = self.policy_function.embedding_decay
                embedding_num = self.policy_function.embedding_num
                eps = self.policy_function.eps

                chosen_num = list()
                embedding_sum = list()
                for i in range(len(self.middle_outputs)):
                    chosen_num.append(len(self.middle_outputs[i]))
                    if len(self.middle_outputs[i]) == 0:
                        embedding_sum.append(torch.zeros(len(prev_embedding_avg[i])))
                    else:
                        embedding_sum.append(torch.stack(self.middle_outputs[i],dim=0).sum(0).to("cpu"))
                embedding_avg = decay*prev_embedding_avg + (1-decay)*torch.stack(embedding_sum, dim=0)
                cluster_size = decay*prev_cluster_size + (1-decay)*torch.tensor(chosen_num)

                n = cluster_size.sum()
                cluster_size_norm = (cluster_size + eps) / (n + embedding_num*eps) * n
                embedding = embedding_avg / cluster_size_norm.unsqueeze(-1)

                self.policy_function.embedding = torch.nn.Parameter(embedding, requires_grad=False)
                self.policy_function.embedding_avg = torch.nn.Parameter(embedding_avg, requires_grad=False)
                self.policy_function.cluster_size = torch.nn.Parameter(cluster_size, requires_grad=False)
                self.policy_function.to(self.device)

                self.middle_outputs = list()
                for i in range(embedding_num):
                    self.middle_outputs.append(list())
        
                loss = self.loss_f(self.actions_prob_history, self.rewards_history, self.beta, self.beta_loss_history)
        else:
            loss = self.loss_f(self.actions_prob_history, self.rewards_history)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if return_loss:
            return loss.to("cpu").detach().numpy()

    def set_rewards(self, reward):
        if self.is_train:
            self.rewards_history.extend(reward)

    def reset_batch(self):
        if self.is_train:
            self.actions_prob_history = list()
            self.rewards_history = list()
            if self.encoder_type == "vq":
                self.embedding_loss_history = list()
                self.beta_loss_history = list()

    def save_model(self, path):
        self.policy_function.to("cpu")
        torch.save(self.policy_function.state_dict(), path)
        
        self.policy_function.to(self.device)