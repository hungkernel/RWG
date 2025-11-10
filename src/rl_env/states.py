
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import gymnasium.spaces as spaces
from src import config
from config import NUM_ACTIONS

class DebateFeedback(BaseModel):
    round: int
    score: float = 0.0
    reasoning: str = ""
    critiques: List[str] = Field(default_factory=list)

class Blackboard(BaseModel):
    session_id: str
    prompt: str 
    outline: List[str] = Field(default_factory=list)
    draft: str = ""
    knowledge_db: List[str] = Field(default_factory=list)
    feedback_history: List[DebateFeedback] = Field(default_factory=list)
    round: int = 0
    max_rounds: int = config.MAX_ROUNDS
    paper_title: str 
    abstract: str

def get_observation_space() -> spaces.Space:
    """Trả về cấu trúc không gian quan sát (O) dạng Vector."""
    embed_dim = config.EMBED_DIM
    
    return spaces.Dict({
        "input_vec": spaces.Box(low=-np.inf, high=np.inf, shape=(embed_dim,)),
        "outline_vec": spaces.Box(low=-np.inf, high=np.inf, shape=(embed_dim,)),
        "document_vec": spaces.Box(low=-np.inf, high=np.inf, shape=(embed_dim,)),
        "knowledge_vec": spaces.Box(low=-np.inf, high=np.inf, shape=(embed_dim,)),
        "agent_state_vec": spaces.Box(low=0, high=1, shape=(NUM_ACTIONS,)),
        "last_feedback_score": spaces.Box(low=0, high=10, shape=(1,)),
        "last_feedback_critique_vec": spaces.Box(low=-np.inf, high=np.inf, shape=(embed_dim,))
    })

def get_action_space() -> spaces.Space:
    """Trả về không gian hành động (A) cho PettingZoo."""
 
    return spaces.Discrete(NUM_ACTIONS)