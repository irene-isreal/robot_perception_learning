# Autonomous Indoor Exploration Robot Using LiDAR, Machine Learning and Reinforcement Learning

A simulation-based autonomous robotics project that explores an indoor environment using simulated LiDAR perception, machine learning, frontier-based navigation, reinforcement learning, and fuzzy-logic obstacle avoidance.

## 1. Project Overview

Autonomous indoor exploration requires a robot to perceive its surroundings, identify obstacles, build an understanding of its environment, and navigate safely through unknown spaces.

This project develops a 2D simulated indoor robot that uses LiDAR distance measurements to perceive its surroundings and explore an environment containing walls, furniture, and other obstacles.

The project investigates multiple approaches to autonomous exploration, including classical machine learning, Q-learning, BFS, A*, and fuzzy-logic safety control.

The main objective is to study how different perception and navigation techniques can be integrated into an autonomous exploration system and evaluated under different starting conditions.

**Project type:** Simulation-based robotics and AI  
**Environment:** 2D indoor environment  
**Language:** Python  
**Primary libraries:** NumPy, Pygame, scikit-learn, Matplotlib, Pandas

## 2. Key Features

- Simulated indoor environment containing walls, furniture, and obstacles.
- 2D LiDAR simulation with ray casting and distance measurements.
- Synthetic LiDAR dataset generation for object classification.
- Custom implementations of K-Nearest Neighbours, Decision Tree, and Random Forest.
- Comparison with scikit-learn machine learning implementations.
- Q-learning-based autonomous exploration.
- Occupancy-grid mapping and frontier-based exploration.
- BFS and A* path-planning algorithms.
- Fuzzy-logic-based obstacle avoidance and safety control.
- Automated multi-seed evaluation of navigation performance.
- Collision, exploration coverage, frontier, and safety metrics.

## 3. System Architecture

The system is divided into the following major components:

1. **Environment Simulation:** Creates the indoor environment and manages robot movement and obstacle interactions.
2. **LiDAR Perception:** Simulates laser scans using ray casting and generates distance measurements.
3. **Machine Learning:** Classifies synthetic LiDAR scan patterns into object categories.
4. **Mapping:** Maintains an occupancy grid representing occupied, free, and unexplored areas.
5. **Navigation:** Selects exploration targets and plans paths using BFS or A*.
6. **Safety Control:** Uses fuzzy logic to evaluate nearby obstacles and override unsafe movement decisions.
7. **Reinforcement Learning:** Implements Q-learning as an alternative exploration approach.
8. **Evaluation:** Measures exploration coverage, collisions, safety overrides, and navigation performance.

### Overall workflow

```text
        Indoor Environment
                |
                v
          LiDAR Simulation
                |
                v
        Distance Measurements
                |
          +-----+------+
          |            |
          v            v
     ML Perception   Occupancy Grid
          |            |
          v            v
     Object Classes  Frontier Detection
                       |
                       v
                BFS / A* Planner
                       |
                       v
                Fuzzy Safety Layer
                       |
                       v
                 Robot Movement
                       |
                       v
              Environment Update
                       |
                       +----> Repeat
```

Q-learning is implemented and evaluated as an additional exploration approach.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| Pygame | 2D simulation and visualization |
| NumPy | Numerical operations and occupancy grids |
| scikit-learn | Machine learning baselines and evaluation |
| Pandas | Dataset and experiment management |
| Matplotlib | Training and evaluation visualizations |
| pytest | Testing |

## 5. LiDAR Simulation

The robot uses simulated LiDAR to measure distances to nearby obstacles.

Ray casting is used to simulate laser beams travelling through the environment. Each ray returns the distance to the nearest detected obstacle, subject to a maximum sensing range.

The interactive simulation uses:

- 180 LiDAR rays.
- Maximum range of 350 pixels.
- A 2D indoor environment with walls and furniture.

These measurements provide the robot with information about its surroundings.

## 6. Machine Learning-Based Perception

A synthetic dataset is generated using simulated LiDAR scans.

The dataset contains four object categories:

- Person
- Box
- Shelf
- Free space

The perception component extracts features from LiDAR measurements, including:

- Normalized distance readings.
- Statistical features.
- Sector-based minimum, mean, and standard deviation.
- Hit masks and distance differences.
- Segmented scan features.

### Models evaluated

- Custom K-Nearest Neighbours.
- scikit-learn K-Nearest Neighbours.
- Custom Decision Tree.
- scikit-learn Decision Tree.
- Custom Random Forest.
- scikit-learn Random Forest.

### Perception results

The models were evaluated using a stratified 75:25 train-test split.

| Model | Accuracy | Macro F1-score |
|---|---:|---:|
| Custom KNN | 67.60% | 68.17% |
| scikit-learn KNN | 67.60% | 68.17% |
| Custom Decision Tree | 78.90% | 78.79% |
| scikit-learn Decision Tree | 78.40% | 78.45% |
| Custom Random Forest | 78.00% | 78.00% |
| scikit-learn Random Forest | 78.60% | 78.41% |

The Decision Tree implementations achieved the highest accuracy in this experiment.

**Dataset limitation:** The dataset is synthetic and generated using isolated object scenarios. These results should not be interpreted as evidence of reliable real-world object detection.

## 7. Autonomous Exploration and Navigation

The navigation component explores unknown areas using an occupancy grid.

Each grid cell represents one of three states:

- `-1`: Occupied.
- `0`: Unknown.
- `1`: Free.

### Frontier-based exploration

A frontier is a free cell adjacent to at least one unknown cell.

The robot identifies frontier cells and selects an exploration target based on information gain and path cost.

This allows the robot to progressively explore unknown parts of the environment.

### Path-planning algorithms

Two path-planning algorithms are implemented:

**Breadth-First Search (BFS)**

- Explores the grid level by level.
- Finds shortest paths in an unweighted grid.
- Used as one of the navigation approaches.

**A* Search**

- Uses path cost and a heuristic to guide its search.
- Uses Manhattan distance as its heuristic.
- Provides an alternative path-planning approach for comparison.

### Fuzzy-logic safety controller

A fuzzy-logic controller evaluates three LiDAR sectors:

- Front
- Left
- Right

Distance measurements are mapped into fuzzy membership categories:

- Near
- Medium
- Far

The controller evaluates these inputs and selects an appropriate movement action.

When the planned forward movement is considered unsafe, the fuzzy controller can override the planner's decision.

The autonomous controller also includes obstacle-escape behaviour and temporary memory of rejected movement cells to reduce repeated attempts.

## 8. Q-Learning-Based Exploration

Q-learning is implemented as an alternative reinforcement-learning approach to autonomous exploration.

The agent learns action preferences using rewards and penalties.

The reward design considers:

- Discovering new cells.
- Revisiting previously explored cells.
- Collision avoidance.
- Movement costs.

A frontier-based Q-learning experiment was trained for 1,200 episodes.

The learned Q-table is saved for later use.

**Note:** Q-learning is an experimental component and is not the primary navigation method used in the final BFS/A* comparison.

## 9. Experimental Evaluation

The final navigation evaluation compares BFS and A* with the fuzzy-logic safety controller enabled.

### Experimental setup

| Parameter | Value |
|---|---|
| Environment | Simulated 2D indoor environment |
| Starting seeds | 42–71 |
| Algorithms | BFS and A* |
| Episodes per algorithm | 30 |
| Total episodes | 60 |
| Steps per episode | 250 |
| Traversable grid cells | 532 |
| Evaluation metrics | Coverage, collisions, frontiers, replanning, safety overrides |

Coverage is calculated using traversable free-space cells rather than the entire grid, which includes obstacles and invalid positions.

### Final results

| Metric | BFS | A* |
|---|---:|---:|
| True free-space coverage | 9.85 ± 2.65% | 9.57 ± 2.06% |
| Visited traversable cells | 52.40 ± 14.11 | 50.93 ± 10.94 |
| Frontiers reached | 4.67 ± 2.11 | 4.27 ± 1.72 |
| Average collisions | 0 | 0 |
| Replans | 14.07 ± 2.57 | 14.73 ± 1.53 |
| Safety override percentage | 4.59 ± 2.30% | 5.01 ± 1.62% |

Values are reported as mean ± standard deviation across 30 episodes per algorithm.

### Observations

- Both navigation approaches completed all 30 evaluated episodes without collisions.
- BFS achieved slightly higher mean exploration coverage than A*.
- BFS also reached slightly more frontier goals on average.
- The difference in coverage is small and does not establish statistical superiority.
- Both algorithms still have room for improvement in exploration efficiency.

These results demonstrate performance within the tested simulation scenarios, not a guarantee of real-world collision avoidance.

## 10. Project Structure

```text
robot_perception_learning/
│
├── main.py
├── config.py
├── requirements.txt
├── README.md
│
├── lidar/
│   ├── simulator.py
│   ├── ray_casting.py
│   └── dataset_generator.py
│
├── perception/
│   ├── knn.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   ├── features.py
│   └── evaluation.py
│
├── reinforcement_learning/
│   ├── environment.py
│   ├── agent.py
│   ├── train.py
│   ├── visualize.py
│   ├── test_policy.py
│   └── evaluate_policy.py
│
├── navigation/
│   ├── grid_planner.py
│   └── autonomous_controller.py
│
├── safety/
│   └── fuzzy_controller.py
│
├── simulation/
│   └── renderer.py
│
├── experiments/
│   ├── train_perception.py
│   ├── test_grid_planner.py
│   ├── test_autonomous_navigation.py
│   ├── evaluate_planners.py
│   ├── test_fuzzy_safety.py
│   ├── test_fuzzy_autonomous_navigation.py
│   ├── diagnose_fuzzy_collision.py
│   └── evaluate_fuzzy_planners.py
│
├── tests/
├── notebooks/
├── results/
└── models/
```

## 11. Installation and Setup

### Prerequisites

- Python 3.10 or later.
- Git.
- VS Code or another Python-compatible editor.

### Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd robot_perception_learning
```

### Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 12. Running the Project

### Run the interactive LiDAR simulation

```bash
python