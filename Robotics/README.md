### Dependencies
- [ROS Melodic](http://wiki.ros.org/melodic/Installation/Ubuntu)
- [catkin_tools](https://catkin-tools.readthedocs.io/en/latest/installing.html)
- [MoveIt](https://moveit.ros.org/install/)

### How to install
Clone our project to a folder of your choice:
```
cd ${PROJECT_PARENT_FOLDER}
git clone https://github.com/BrunoGeorgevich/Team_Blue.git
```

Create a new empty workspace called ```tictactoe_ws``` inside your home folder. 
```
cd ~
mkdir tictactoe_ws
```

Create a symbolic link from our Team_Blue/src folder to your new workspace.
Then, build it:
```
ln -s ${PROJECT_PARENT_FOLDER}/Team_Blue/Robotics/src ~/tictactoe_ws/src
cd ~/tictactoe_ws/src
catkin build
```

If some dependency is missing, install it with rosdep and build workspace again:
```
rosdep install -y --from-paths . --ignore-src --rosdistro melodic
catkin build
```

Add workspace activation command to your .bashrc:
```
echo "export LC_NUMERIC="en_US.UTF-8"" >> ~/.bashrc
echo "source ~/tictactoe_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

Update package auto-completion:
```
rospack profile
```

### How to run
Run the MoveIT demo:
```
roslaunch vp6242b_moveit_config demo.launch 
```

