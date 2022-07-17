# Laboratorio

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]




<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Physical human-robot collaboration is increasingly required in many contexts (such as industrial and rehabilitation applications). The robot needs to interact with the human to perform the target task while relieving the user from the workload. To do that, the robot should be able to recognize the human’s intentions and guarantee the safe and adaptive behavior along the intended motion directions. These advanced robot-control strategies are particularly demanded in the industrial field, where the operator guides the robot manually to manipulate heavy parts (e.g., while teaching a specific task). With this aim, this repository contains the code to implement a Q-Learning-based Model Predictive Variable Impedance Control (Q-LMPVIC) to assist the operators in physical human-robot collaboration (pHRC) tasks. A Cartesian impedance control loop is designed to implement a decoupled compliant robot dynamics. The impedance control parameters (i.e., setpoint and damping parameters) are then optimized online in order to maximize the performance of the pHRC. For this purpose, an ensemble of neural networks is designed to learn the modeling of the human-robot interaction dynamics while capturing the associated uncertainties. The derived modeling is then exploited by the MPC, enhanced with the stability guarantees by means of Lyapunov constraints. The MPC is solved by making use of a Q-Learning method that, in its online implementation, uses an actor-critic algorithm to approximate the exact solution. The Q-learning method provides an accurate and highly efficient solution (in terms of computational time and resources). 

The proposed control strategy has been validated through a comparison with the [Model-Based Reinforcement Learning Variable Impedance Control](https://doi.org/10.1007/s10846-020-01183-3) (MBRLVIC), a previously developed method available in literature. Both these algorithms are developed in the present repository.

### Content

In folder [franka_example_controllers][frankaexamplecontrollers] a set of example controllers for controlling the robot via ROS are implemented. Among them, there are the two low-level cartesian impedance controllers to run the Q-LMPVIC ([cartesian_impedance_QLMPC_controller.h][impedance_QLMPC]) and the MBRLVIC ([cartesian_impedance_MBRL_controller.h][impedance_MBRL]) updating strategies.

These stretegies are implemented in folder [Updating stategies][updatingstrategies] ([Q_LMPC_simplified_revised.py][QLMPC]) ([MBRL_controller_confronto.py][MBRL]). The other files refer to past versions of the abovementioned codes. In the same folder are available also the models of the pre-trained artificial neural networks and the data used to normalize them.

### Built With

* [libfranka](https://github.com/frankaemika/libfranka)
* [franka_ros](https://github.com/frankaemika/franka_ros)


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and run the controller follow these simple steps.

### Prerequisites

The proposed approach has been validated using a Franka EMIKA panda robot as a test platform. Thus, it is required the installation of the Franka Control Interface to run the proposed code. See the [Franka Control Interface (FCI) documentation][fci-docs] for more information.

### Installation

1. Create a catkin workspace in a directory of your choice. Open terminal and navigate to src folder in your catkin workspace:
   ```sh
   cd ~/catkin_ws/src
   ```
2. Clone the repository using:
   ```sh
   git clone https://github.com/Andrea8Testa/Laboratorio.git
   ```
The present repository contains two main folders. [franka_example_controllers][frankaexamplecontrollers] must be used to replace the homonymous folder in [franka_ros](https://github.com/frankaemika/franka_ros). [Updating stategies][updatingstrategies] instead, must be kept inside the package.

3. Build the package:
   ```sh
    cd ~/catkin_ws 
    catkin_make
   ```

<!-- USAGE EXAMPLES -->
### Run the algorithm

First of all, activate the variable impedance controller running the following command inside the folder [franka_example_controllers/include/franka_example_controllers](https://github.com/Andrea8Testa/Laboratorio/tree/main/franka_example_controllers/include/franka_example_controllers):
```
./cartesian_impedance_QLMPC_controller
```
Then, move to folder [Updating strategies](https://github.com/Andrea8Testa/Laboratorio/tree/main/Updating%20strategies) and run the QLMPC by the command:
```
python Q_LMPC_simplified_revised.py
```
Once started, the robot end-effector should stand still  in the initializiation position until the operator touches it. Then, the end-effector is expected to move to minimize the human-interaction force.

### ANNs settings

We set the QLMPC to start with pre-trained artificial neural networks. Comment the following lines if you want to perform your own training.  
- [model_approximator["NN0"].load_state_dict(torch.load(PATH_model0))](https://github.com/Andrea8Testa/Laboratorio/blob/main/Updating%20strategies/Q_LMPC_simplified_revised.py#L979)
- [model_approximator["NN1"].load_state_dict(torch.load(PATH_model1))](https://github.com/Andrea8Testa/Laboratorio/blob/main/Updating%20strategies/Q_LMPC_simplified_revised.py#L980)
- [actor.load_state_dict(torch.load(PATH_actor))](https://github.com/Andrea8Testa/Laboratorio/blob/main/Updating%20strategies/Q_LMPC_simplified_revised.py#L988)
- [critic.load_state_dict(torch.load(PATH_critic))](https://github.com/Andrea8Testa/Laboratorio/blob/main/Updating%20strategies/Q_LMPC_simplified_revised.py#L989)

The ANNs are initialized with random weights in range `(1e-2, 1e-1)`. The QLMPC works at a frequency of `6` Hz, the training buffer size is `5`, so the ANNs are updated every `0.83` seconds. 

#### Hyperparameters

|    Model approximator   |     Value     |
| ----------------------- | ------------- |
| Number of hidden layers |       5       |
| Number of hidden units  |      512      |
| Size of the ensemble    |       2       |
| Activation function     |     ReLu      |
|Dropout probability inner layers|  0.2   |
|Dropout probability last layer|   0.1    |
|Learning rate|   1e-3    |

|          Actor          |     Value     |
| ----------------------- | ------------- |
| Number of hidden layers |       3       |
| Number of hidden units  |      64       |
| Activation function inner layers| ReLu  |
| Activation function last layer |  Tanh  |
|Dropout probability      |      0.1      |
|Learning rate for an interaction force < 0.1 N|   5e-5    |
|Learning rate for an interaction force < 0.5 N|   8e-5    |
|Learning rate otherwise|   1e-4    |

|         Critic          |     Value     |
| ----------------------- | ------------- |
| Number of hidden layers |       3       |
| Number of hidden units  |      128      |
| Activation function     |     ReLu      |
|Dropout probability      |      0.5      |
|Learning rate|   1e-3    |

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/Andrea8Testa/Laboratorio/issues) for a list of proposed features (and known issues).

<!-- CONTACT -->
## Contact

[Andrea Testa][linkedin-url] - email andrea3.testa@gmail.com

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

The work has been developed within the project ASSASSINN, funded from H2020 CleanSky 2 under grant agreement n. 886977.

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Andrea8Testa/Laboratorio.svg?style=for-the-badge
[contributors-url]: https://github.com/Andrea8Testa/Laboratorio/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Andrea8Testa/Laboratorio.svg?style=for-the-badge
[forks-url]: https://github.com/Andrea8Testa/Laboratorio/network/members
[stars-shield]: https://img.shields.io/github/stars/Andrea8Testa/Laboratorio.svg?style=for-the-badge
[stars-url]: https://github.com/Andrea8Testa/Laboratorio/stargazers
[issues-shield]: https://img.shields.io/github/issues/Andrea8Testa/Laboratorio.svg?style=for-the-badge
[issues-url]: https://github.com/Andrea8Testa/Laboratorio/issues
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/andrea-testa-b0ba8714b

[fci-docs]: https://frankaemika.github.io/docs
[updatingstrategies]: https://github.com/Andrea8Testa/Laboratorio/tree/main/Updating%20strategies
[frankaexamplecontrollers]: https://github.com/Andrea8Testa/Laboratorio/tree/main/franka_example_controllers
[impedance_QLMPC]: https://github.com/Andrea8Testa/Laboratorio/blob/main/franka_example_controllers/include/franka_example_controllers/cartesian_impedance_QLMPC_controller.h
[impedance_MBRL]: https://github.com/Andrea8Testa/Laboratorio/blob/main/franka_example_controllers/include/franka_example_controllers/cartesian_impedance_MBRL_controller.h
[QLMPC]: https://github.com/Andrea8Testa/Laboratorio/blob/main/Updating%20strategies/Q_LMPC_simplified_revised.py
[MBRL]: https://github.com/Andrea8Testa/Laboratorio/blob/main/Updating%20strategies/MBRL_controller_confronto.py
