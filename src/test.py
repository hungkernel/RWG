from agent.model import WorkerLLM

prompt = "Giải thích ngắn gọn về Tree-Wasserstein Distance."
result = WorkerLLM().generate(prompt)
print(result)
