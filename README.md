# Tetris ML Project
For this project, I developed a reinforcement learning agent capable of playing Tetris using a neural network to approximate Q-values. The agent was trained using temporal difference learning and self-play, where it simulated thousands of games to improve performance over time. I engineered a custom input vector that encoded both the game state and possible placements of Tetris pieces, and designed a reward function to guide the agent’s decision-making based on outcomes like line clears and board stability. To stabilize learning, I implemented a replay buffer that stored past experiences for batch training, and incorporated exploration techniques to prevent the agent from getting stuck in suboptimal policies. The result was an agent that learned increasingly efficient gameplay strategies through iterative training and evaluation cycles.

## How to run:<br/>
javac -cp "./*:." @tetris.srcs<br/>
java -cp "./*:." edu.bu.tetris.Main
