import textarena as ta

# Initialize agents
agents = {
    0: ta.agents.OpenRouterAgent(model_name="llama-3.1-8b"),
    1: ta.agents.OpenRouterAgent(model_name="llama-3.1-8b"),
}

    
env_list =[
    "SpellingBee-v0",
    "Poker-v0",
    "SpiteAndMalice-v0",
    "Stratego-v0",
    "Tak-v0",
    "TruthAndDeception-v0",
    "UltimateTicTacToe-v0",
    "WordChains-v0",
    "TicTacToe-v0",
    "Breakthrough-v0",
    "Checkers-v0",
    "KuhnPoker-v0",
    "LetterAuction-v0",
    "MemoryGame-v0",
    "Nim-v0",
    "Othello-v0",
    "PigDice-v0",
    "SimpleBlindAuction-v0",
    "Snake-v0",
    "SecretMafia-v0",
    # "WildTicTacToe-v0",
    # "ReverseTicTacToe-v0",
    # "RandomizedTicTacToe-v0",
    # "QuantumTicTacToe-v0",
]


for env_id in env_list:
    print(f"Playing {env_id}...")
    with open("textarena.log", "a") as log_file:
        log_file.write(f"Playing {env_id}...\n")
    env = ta.make(env_id=env_id)
    env = ta.wrappers.LLMObservationWrapper(env=env)
    # Optional render wrapper 
    env = ta.wrappers.SimpleRenderWrapper(
        env=env,
        player_names={0: "GPT-4o-mini", 1: "claude-3.5-haiku"},
    )

    env.reset(num_players=len(agents))
    done = False
    while not done:
        player_id, observation = env.get_observation()
        action = agents[player_id](observation)
        done, info = env.step(action=action)
    rewards = env.close()
    with open("textarena.log", "a") as log_file:
        log_file.write(f"Rewards for {env_id}: {rewards}\n")
    print(f'For {env_id} the rewards are {rewards}')