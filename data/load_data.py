# search_action_openalex_env.py
import os
import requests
import textwrap
from typing import List, Optional
from rl_env.graph import GraphEnviroment
from agent.action import BaseAction
# Lấy URL từ biến môi trường (mặc định về OpenAlex)
OPENALEX_BASE = os.getenv("OPENALEX_BASE")

# Viết ở đây sử dụng openalex




def search_paper (search_query: str) -> List[dict]:
    """Tìm kiếm OpenAlex và trả về danh sách các công trình."""
    papers = List[dict]
    '''
    Format yêu cầu: 
        title : str
        abstract : str
        introduction: str
        citations : int
    '''
    return papers
