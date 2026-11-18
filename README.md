An interactive Snake game used primarily to simulate different AI agents
Implemented Agents:

Baseline | Simple Reflex Agent | Greedy Manhattan-distance heuristic choosing among immediate safe moves. |
BFS | Uninformed Search | Breadth-First Search with dynamic tail-position simulation and flood-fill fallback. |
A* | Informed Search | Manhattan-heuristic shortest-path search with body-segment time indexing and safe fallback. |
CSP | Constraint Satisfaction | 4-step lookahead backtracking solver with forward checking and global tail/flood-fill safety constraints. |
GA (1-Ply & 2-Ply) | Evolutionary / Utility-Based | Numba-accelerated linear utility evaluator trained via a Genetic Algorithm across 6 board features (food distance, free area ratio, tail reachability, local hazards, and consumption reward). |
Minimax | Adversarial Search | Depth-limited Minimax with Alpha-Beta pruning for 2-snake competitive territory and pellet control. |

Features:
Single Player
Agent Simulation
1 V AI
AI v AI

installation:
1.Make sure you have all requirements listed in requirements.txt.
2.Extract repository and navigate to root, in terminal write: 
  cd snake-ai-benchmark
3.Set up virtual environment, in terminal write: 
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  venv\Scripts\activate.bat
4.Install dependancies, in terminal write:
  python -m pip install --upgrade pip
  pip install -r requirements.txt
5.Run game, in terminal write:
  python main.py