import torch
import numpy as np
from src.controller.actor import Actor
from src.controller.critic import Critic
from src.agent.model import WorkerLLM
class Agent:
    def __init__(self, agent_id: str, actor: Actor, critic: Critic, llm: WorkerLLM):
        self.id = agent_id
        self.actor = actor
        self.critic = critic
        self.llm = llm

    def preprocess_obs(self, obs_dict: dict) -> torch.Tensor:
        """Hãy viết hàm này để xử lí các góc nhìn """

        vec = np.concatenate([
            obs_dict["input_vec"],
            obs_dict["outline_vec"],
            obs_dict["knowledge_vec"],
            obs_dict["document_vec"],
            obs_dict["agent_state_vec"],
            obs_dict["last_feedback_score"],
            obs_dict["last_feedback_critique_vec"]
        ])
        return torch.tensor(vec, dtype=torch.float32).to(self.actor.device)

    def get_action_and_value(self, obs_dict: dict, action: int = None):
        x_flat = self.preprocess_obs(obs_dict).unsqueeze(0)
        actor_output = self.actor(x_flat)
        if isinstance(actor_output, tuple):
            logits = actor_output[0]
        else:
            logits = actor_output
        value = self.critic(x_flat)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value