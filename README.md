# CARLA-Reinforcement learning
In this repository Deep Q-Learning is used for self-driving vehicle in CARLA environment. The algorithm is implemented in a quite simple environment with few surrounding vehicle. An example of the result can be seen bellow. Note that the agent requires to be trained longer than the figure provided with more obstacles on the road. 
![Figure_1](https://user-images.githubusercontent.com/51369142/86182178-5e64bb80-bb27-11ea-870e-6fc0f4048408.png)

## Installation
Clone the repository `
git clone https://github.com/shayantaherian/Reinforcement-Learning-CARLA/.git
`

Install the requirement `
requirement.txt
`
#### CARLA Installation
Download Carla you can just download the compiled version from [here](https://carla.org/). Note that it is reuiqred to download the stable version of the simulator

## Taining

First run the carla server  `CarlaUE4.sh ` from the save directory

Then run `
python Main.py
`
.To test the results run `
python Test.py
`
