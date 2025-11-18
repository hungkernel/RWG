"""
Main training script for the RWG (Related Work Generation) multi-agent RL system.
"""
import torch
import numpy as np
from src.rl_env.graph import GraphEnviroment
from src import config

def main():
    """Main training loop."""
    print("Initializing RWG Environment...")

    # Create environment
    env = GraphEnviroment()

    # Example task prompt
    task_prompt = "Write a Related Work section about multi-agent reinforcement learning for academic writing."

    print(f"\nTask: {task_prompt}\n")

    # Reset environment
    observations, infos = env.reset(
        seed=42,
        options={
            "prompt": task_prompt,
            "paper_title": "Multi-Agent RL for Academic Writing",
            "abstract": "This paper explores the use of multi-agent reinforcement learning..."
        }
    )

    print(f"Environment reset. Number of agents: {len(env.agents)}")
    print(f"Observation space keys: {list(observations[env.agents[0]].keys())}")

    # Run a few episodes
    max_episodes = 1
    for episode in range(max_episodes):
        print(f"\n=== Episode {episode + 1} ===")

        done = False
        step_count = 0

        while not done and step_count < config.MAX_ROUNDS:
            print(f"\n--- Step {step_count + 1} ---")

            # Sample actions for all agents
            actions = {}
            for agent_id in env.agents:
                agent = env.agent_nodes[agent_id]
                obs = observations[agent_id]

                # Get action from agent
                action, log_prob, value = agent.get_action_and_value(obs)
                actions[agent_id] = action

                from src.agent.action import AgentAction
                action_name = AgentAction(action).name
                print(f"{agent_id}: Action {action} ({action_name}) - log_prob: {log_prob.item():.4f}, value: {value.item():.4f}")

            # Step environment
            observations, rewards, terminations, truncations, infos = env.step(actions)

            # Print rewards
            print("Rewards:", {aid: f"{r:.4f}" for aid, r in rewards.items()})

            # Check if done
            done = all(terminations.values()) or all(truncations.values())
            step_count += 1

            # Print current state
            if env.state:
                print(f"Round: {env.state.round}")
                print(f"Outline items: {len(env.state.outline)}")
                print(f"Knowledge DB items: {len(env.state.knowledge_db)}")
                print(f"Draft length: {len(env.state.draft)} chars")

        print(f"\nEpisode {episode + 1} completed after {step_count} steps")
        if env.state:
            print(f"\nFinal Draft Preview (first 500 chars):")
            print(env.state.draft[:500] + "..." if len(env.state.draft) > 500 else env.state.draft)

    print("\nTraining completed!")

if __name__ == "__main__":
    main()

