import torch
import numpy as np

# 方策関数
class PolicyFunction(torch.nn.Module):
    def __init__(
        self, num_states, num_traffic_lights, num_actions, num_layers=1, 
        num_hidden_units=128, temperature=1.0, noise=0.0, encoder_type="fc",
        embedding_num=5):
        
        super().__init__()
        self.num_states = num_states
        self.num_traffic_lights = num_traffic_lights
        self.num_actions = num_actions
        self.temperature = temperature
        self.noise = noise
        self.encoder_type = encoder_type
        self.embedding_num = embedding_num

        if self.encoder_type == "fc":
            self.fc_first = torch.nn.Linear(self.num_states, num_hidden_units)
            self.encoder = self.fc_encoder
        elif self.encoder_type == "lstm":
            self.lstm = torch.nn.LSTM(self.num_states, num_hidden_units, 1, batch_first=True)
            self.fc_first = torch.nn.Linear(num_hidden_units, num_hidden_units)
            self.encoder = self.lstm_encoder
        elif self.encoder_type == "vq":
            self.fc_first = torch.nn.Linear(self.num_states, num_hidden_units)
            self.embedding = torch.nn.Embedding(self.embedding_num, num_hidden_units)
            self.encoder = self.vq_encoder
        
        
        self.fc_last_layers = list()
        for i in range(num_traffic_lights):
            self.fc_last_layers.append(torch.nn.Linear(num_hidden_units, self.num_actions))

        self.fc_layers = list()
        self.num_layers = num_layers
        for i in range(num_layers):
            self.fc_layers.append(torch.nn.Linear(num_hidden_units, num_hidden_units))

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)
    
    def fc_encoder(self, inputs):
        x = self.fc_first(inputs)
        x = self.relu(x)
        for i in range(self.num_layers):
            x = self.fc_layers[i](x)
            x = self.relu(x)
        
        if self.training:
            x = x + torch.normal(torch.zeros(x.shape[-1]), torch.ones(x.shape[-1])*self.noise)
        
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
            x = x + torch.normal(torch.zeros(x.shape[-1]), torch.ones(x.shape[-1])*self.noise)
        
        return x
    
    def vq_encoder(self, inputs):
        x = self.fc_first(inputs)
        x = self.relu(x)
        for i in range(self.num_layers):
            x = self.fc_layers[i](x)
            x = self.relu(x)
        
        embedding_weights = self.embedding.weight.detach()
        vector = x.detach()
        embedding_idx = (vector - embedding_weights).pow(2).sum(-1).argmin(-1)

        quantize = x + (embedding_weights[embedding_idx] - vector)
        if self.training:
            embedding_loss = (vector - self.embedding(embedding_idx)).pow(2).sum(-1)
            beta_loss = (x - embedding_weights[embedding_idx]).pow(2).sum(-1)
            return quantize, embedding_loss, beta_loss
        else:
            return quantize

    def forward(self, inputs):
        if self.encoder_type == "vq" and self.training:
            x, embedding_loss, beta_loss = self.encoder(inputs)
        else:
            x = self.encoder(inputs)

        last_outputs = list()
        for i in range(self.num_traffic_lights):
            last_x = self.fc_last_layers[i](x)
            last_x = self.softmax(last_x/self.temperature).unsqueeze(0)
            last_outputs.append(last_x)
        outputs = torch.cat(last_outputs, dim=0)
        
        if self.encoder_type == "vq" and self.training:
            return outputs, embedding_loss, beta_loss
        else:
            return outputs

# 損失関数(期待収益のマイナス符号)
class PolicyGradientLossWithREINFORCE(torch.nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, actions_prob_history, rewards_history, embedding_loss_history=None, beta=0.25, beta_loss_history=None):
        ave_rewards = np.mean(rewards_history)
        loss = 0
        for i in range(len(actions_prob_history)):
            chosen_action_prob = actions_prob_history[i]
            # 最大化する関数のマイナス
            loss = loss - torch.log(chosen_action_prob) * (rewards_history[i] - ave_rewards)
        
        loss = loss / len(actions_prob_history)
        
        if embedding_loss_history is not None and beta_loss_history is not None:
            vq_loss = 0
            for i in range(len(embedding_loss_history)):
                # 埋め込み表現更新のための損失
                vq_loss = vq_loss + embedding_loss_history[i]
                # 埋め込み前のベクトルが埋め込み表現から一気にずれ過ぎないようにする損失
                vq_loss = vq_loss + beta*beta_loss_history[i]
            vq_loss = vq_loss / len(embedding_loss_history)
            loss = loss + vq_loss

        return loss


# エージェント(方策関数を学習し、行動を決定する)
class Agent():
    def __init__(
        self, num_states, num_traffic_lights, num_actions, num_layers, num_hidden_units, 
        temperature, noise, encoder_type, is_train=True, lr=3e-5, decay_rate=0.01, use_gpu=False, model_path=None
        ):
        
        self.num_states = num_states
        self.num_traffic_lights = num_traffic_lights
        self.num_actions = num_actions
        self.lr = lr
        self.encoder_type = encoder_type
        self.is_train = is_train
        self.use_gpu = use_gpu

        self.policy_function = PolicyFunction(
            self.num_states, self.num_traffic_lights, self.num_actions, num_layers, 
            num_hidden_units, temperature, noise, encoder_type)
        if model_path:
            self.policy_function.load_state_dict(torch.load(model_path))
        if self.use_gpu:
            device = torch.device("cuda")
            self.policy_function.to(device)

        if self.is_train:
            self.actions_prob_history = list()
            self.rewards_history = list()
            if self.encoder_type == "vq":
                self.embedding_loss_history = list()
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
        obs = torch.tensor(obs, dtype=torch.float32)
        if self.use_gpu:
            device = torch.device("cuda")
            obs = obs.to(device)

        if self.is_train:
            if self.encoder_type == "vq":
                actions_prob, embedding_loss, beta_loss = self.policy_function(obs)
            else:
                actions_prob = self.policy_function(obs)
        else:
            with torch.no_grad():
                actions_prob = self.policy_function(obs)
        
        actions_prob_numpy = actions_prob.to("cpu").detach().numpy()
        chosen_actions = list()
        for i in range(self.num_traffic_lights):
            chosen_actions.append(np.random.choice(a=np.arange(4), size=1, replace=True, p=actions_prob_numpy[i])[0])

        if self.is_train:
            for i in range(self.num_traffic_lights):
                self.actions_prob_history.append(actions_prob[i, chosen_actions[i]])
            if self.encoder_type == "vq":
                self.embedding_loss_history.append(embedding_loss)
                self.beta_loss_history.append(beta_loss)
        
        return chosen_actions

    def train(self, return_loss=False):
        if self.encoder_type == "vq":
            loss = self.loss_f(self.actions_prob_history, self.rewards_history, self.embedding_loss_history, 0.25, self.beta_loss_history)
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

        if self.use_gpu:
            device = torch.device("cuda")
            self.policy_function.to(device)