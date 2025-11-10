
PLAN_SYNTHESIS_TEMPLATE = """
You are a meticulous scientific researcher and planner. Your task is to create a detailed, well-structured outline
for a "Related Work" section of an academic paper based on the provided inputs.
Base your plan on the following inputs:
1.  **Main Goal:** {task}
2.  **Retrieved Knowledge (Summary):** {knowledge_summary}
3.  **Existing Drafts (Summary):** {draft_summary}

Instructions:
- Analyze all inputs to understand the goal, available information, and what is already written.
- Generate a logical and comprehensive outline.
- The outline should guide the writing of a high-quality "Related Work" section.

Output Format:
- Return ONLY a list of topics.
- Each topic must be on a new line.

Example:
Topic 1: Introduction to [Main Research Area]
Topic 2: Foundational Methodologies in [Field]
Topic 3: State-of-the-art Approaches for [Specific Problem]
Topic 4: Comparative Analysis of [Method A] vs [Method B]
Topic 5: Identified Research Gaps and Unsolved Challenges
"""


SEARCH_QUERY_TEMPLATE = """
You are an expert information retrieval specialist.
Based on the following user input, generate one concise and effective search query suitable for finding 
relevant academic literature.

Input:
** Task: "{task}"
** paper_title: "{paper_title}"
** abstract: "{abstract}"  
Search Query:
"""

WRITE_DRAFT_TEMPLATE = """
You are a formal academic writer.
Your task is to write a single, coherent paragraph or section based on the provided outline and context.

**Main Goal :** {task}
**Section to Write :** {outline}

**Retrieved Context:**
---
{context}
---

**Instructions:**
1.  Write a clear, concise, and academic draft for the "Section to Write".
2.  You MUST base your writing **only** on the information provided in the "Retrieved Context".
3.  Do NOT invent information or use any external knowledge.
4.  If the context is insufficient to write the section, state: "Insufficient context to write this section."

Draft:
"""

CRITIQUE_PROMPT_TEMPLATE = """
You are a demanding and highly critical scientific reviewer.
Your goal is to provide a harsh but constructive critique of the following text.
 Focus on logical flaws, weak arguments, lack of evidence, poor structure, and unclear writing.
Be specific and actionable.

**Text to Critique:**
---
{text_to_evaluate}
---
Critique:
"""

JUDGE_PROMPT_TEMPLATE = """
You are an objective evaluation agent.
Your task is to review the original text and its critique, then provide a final score and a brief justification.

**Content:**
{text_to_evaluate}

**Critique:**
{critique_1}

**Task:**
Provide a final score from 0.0 (worst) to 10.0 (best) and a brief reasoning.
You MUST respond in JSON format ONLY.

{{"score": X.X, "reasoning": "..."}}
"""