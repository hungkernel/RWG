# policy_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any, TYPE_CHECKING
import json
from src.prompts import PLAN_SYNTHESIS_TEMPLATE, SEARCH_QUERY_TEMPLATE, WRITE_DRAFT_TEMPLATE, CRITIQUE_PROMPT_TEMPLATE,JUDGE_PROMPT_TEMPLATE
from enum import IntEnum

# Avoid circular import - import at runtime when needed
if TYPE_CHECKING:
    from src.rl_env.graph import GraphEnviroment
    from src.rl_env.states import DebateFeedback
# Note: search_paper should be implemented in data/load_data.py
# For now, using a placeholder
try:
    from data.load_data import search_paper
except ImportError:
    # Placeholder function if data.load_data is not available
    def search_paper(query: str):
        return []

class AgentAction(IntEnum):
    PLAN = 0        # Lập dàn ý
    SEARCH = 1      # Tìm kiếm (trên knowledge_db)
    WRITE = 2       # Viết một đoạn (draft) mới
    REVIEW = 3      # Đánh giá (hành động này kích hoạt Debate)
    WAIT = 4       # Chờ (quan trọng để tránh xung đột)

# Đảm bảo tính trừu tượng hóa
class BaseAction:
    def execute(self, env: "GraphEnviroment", agent_id: str) -> Any:
        raise NotImplementedError

class Plan(BaseAction):
    def execute(self, env: "GraphEnviroment", agent_id: str) -> str:
        plan_prompt = PLAN_SYNTHESIS_TEMPLATE.format(
            task=env.state.prompt,
            knowledge_summary=" ".join(env.state.knowledge_db),
            draft_summary=env.state.draft
        )
        new_outline = env.llm.generate(plan_prompt)
        env.state.outline.append(new_outline)

        return new_outline

class Search(BaseAction):
    def execute(self, env: "GraphEnviroment", agent_id: str) -> str:
        query_prompt = f"Từ input sau: '{env.state.prompt}', hãy tạo một search query."
        search_query = env.llm.generate(query_prompt)
        papers = search_paper(search_query)
        if not papers:
            msg = "Không tìm thấy tài liệu nào cho truy vấn: " + search_query
            return msg
        env.state.knowledge_db.append(papers)
        return papers

class Write(BaseAction):
    def execute(self, env: "GraphEnviroment", agent_id: str) -> str:
        context = " ".join(env.state.knowledge_db)
        draft_prompt = WRITE_DRAFT_TEMPLATE.format(
            task=env.state.prompt,
            outline=env.state.outline[-1] if env.state.outline else "Outline is not exits",
            context=context
        )
        new_draft = env.llm.generate(draft_prompt)
        env.state.draft += "\n" + new_draft
        return new_draft

class Review(BaseAction):
    def execute(self, env: "GraphEnviroment", agent_id: str) -> "DebateFeedback":
        # Import here to avoid circular import
        from src.rl_env.states import DebateFeedback

        text_to_evaluate = env.state.draft # Đánh giá toàn bộ draft

        critique_prompt = CRITIQUE_PROMPT_TEMPLATE.format(
            text_to_evaluate=text_to_evaluate
        )
        critique = env.llm.generate(critique_prompt)

        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
            text_to_evaluate=text_to_evaluate,
            critique_1=critique
        )
        judge_response_str = env.llm.generate(judge_prompt)

        try:
            if "```json" in judge_response_str:
                judge_response_str = judge_response_str.split("```json")[1].split("```")[0]
            data = json.loads(judge_response_str)
            score = float(data["score"])
            reasoning = data["reasoning"]
        except Exception:
            score = 0.0
            reasoning = "JSON error"

        return DebateFeedback(
            round=env.state.round,
            score=score,
            reasoning=reasoning,
            critiques=[critique]
        )

class Wait(BaseAction):
    def execute(self, env: "GraphEnviroment", agent_id: str) -> None:
        return None


ACTION_DISPATCHER: Dict[AgentAction, BaseAction] = {
    AgentAction.PLAN: Plan(),
    AgentAction.SEARCH: Search(),
    AgentAction.WRITE: Write(),
    AgentAction.REVIEW: Review(),
    AgentAction.WAIT: Wait(),
}

# Export NUM_ACTIONS for use in other modules
NUM_ACTIONS = len(AgentAction)
