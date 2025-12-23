# AutoRace_ROS_Competition_2025

Установка:

cd ros2_ws/src

git clone https://github.com/Nek1tt/ROS_autorace_competition_2025.git

colcon build

source install/setup.bash

Терминал 1:
ros2 launch robot_bringup autorace_2025.launch.py

Терминал 2:
ros2 launch autorace_core_ROSchupepiki autorace_core.launch.py

Терминал 3:
ros2 run referee_console mission_autorace_2025_referee
