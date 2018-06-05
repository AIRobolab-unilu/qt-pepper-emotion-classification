# QT and Pepper emotion classification

This repository aims to perform emotion classification with the QT robot and Pepper in order to have a better understanding of the human-robot interacton context. 

## Getting Started

You should have installed on your computer [**ROS Kinetic**](http://wiki.ros.org/kinetic/Installation) and **Python 2.7**(which is preinstalled on Ubuntu).

Clone this git repository on your local machine (QT or your computer) in you ros catkin workspace.

```
~/catkin_ws$ git clone --recursive https://github.com/AIRobolab-unilu/qt-pepper-emotion-classification.git
```

In order to be able to use those programs on your computer or on the robots, you will have to make some modifications in your **.bashrc** files in the /home directory.

First open the .bashrc.

`~$ sudo nano .bashrc`

 Then add these lines in it.
 
```
########################
# ROS network connections

#MY IP Playground
MY_IP=10.212.232.16

#QT IP
#ROBOT_IP=10.212.232.14

#Pepper IP
#ROBOT_IP=10.212.232.100

#export ROS_MASTER_URI=http://$ROBOT_IP:11311/
#export ROS_MASTER_URI=http://$MY_IP:11311/

export ROS_IP=$MY_IP
```
Replace adress in MY_IP with your actual IP adress. Check it with the `ifconfig` command, for me it was in the **wlp50** section in **inet addr** (2nd line) :

```
wlp5s0    Link encap:Ethernet  HWaddr 24:fd:52:7a:04:cb  
          inet addr:10.212.232.13  Bcast:10.212.233.255  Mask:255.255.254.0
          inet6 addr: 2001:a18:a:2fec:768e:5ad7:3234:81b8/128 Scope:Global
          inet6 addr: fe80::5b75:af:ce61:d31/64 Scope:Link
          UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
          RX packets:81063 errors:0 dropped:0 overruns:0 frame:0
          TX packets:44139 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000 
          RX bytes:81344563 (81.3 MB)  TX bytes:5680750 (5.6 MB)
```

You will also have to **check and replace the IP addresses of both robots** by the same process using `ifconfig` on them.

- If you are running the roscore on your computer just uncomment this line `#export ROS_MASTER_URI=http://$MY_IP:11311/`

- If you are running the roscore on QT uncomment these 2 lines : 

`#ROBOT_IP=10.212.232.14`

`#export ROS_MASTER_URI=http://$ROBOT_IP:11311/`

- If you are running the roscore on Pepper uncomment these 2 lines : 

`#ROBOT_IP=10.212.232.100`

`#export ROS_MASTER_URI=http://$ROBOT_IP:11311/`

## The face_classification package

This package is used to perform face emotion recognition with the QT robot and Pepper.

With QT, the package includes a program that can mimic the emotion shown by the user QT is interacting with and seeing with its camera, a sort of **face mirroring**. It can also say outloud the emotion it is seeing.

### Prerequisites 
You need to install the following python libraries before running the package. To do that, you can use pip (to install it, write in the terminal `sudo apt-get install python-pip`)

```
pip install numpy scipy scikit-learn pillow tensorflow pandas h5py opencv-python keras==2.0.9 statistics pyyaml pyparsing cycler matplotlib Flask
pip install panda
```
Please note that you need to install **keras version 2.0.9** and not another one.

You might also need to install [tensorflow with virtualenv](https://www.tensorflow.org/install/install_linux#InstallingVirtualenv) if the pip version creates errors when the package is run.

If you don't have a node publishing the camera images in a specific topic, you may also have to install a ROS package that can open the camera ([cv_camera](https://github.com/OTL/cv_camera)).

### Installing and running

To run the package, you have different options using the launch files :

*  Run the program directly onboard the qt or any computer but without relying on topic to transmit the images

`roslaunch face_classification face_classification_basic.launch`

*   While on QT, run the program directly as previously but also activates the face mirroring and the speech module so qt can say the emotion it is recognizing

`roslaunch face_classification face_classification_qt_basic.launch`

*  Run the program directly onboard the qt or any computer using topic to transmit the images

`roslaunch face_classification face_classification.launch`

*  While on QT, run the program using topic to transmit the images but activates the face mirroring and the speech module so qt can say the emotion it is recognizing

`roslaunch face_classification face_classification_qt.launch`

Also, you can use these options when you launch the launch files to specify the models used or the camera used:

* detection_model_path for the face detection model
* emotion_model_path for the emotion recognition model
* input_camera_topic for the camera from which the images are grabbed
* output_camera_topic for the resulting images after classification

Here is an example :
`roslaunch face_classification face_classification_qt.launch image_topic_output:=/face_detection/image_raw image_topic_input:=/your_camera/image_raw`

To run the camera to feed the image topic, you can run in another terminal the command : `rosrun cv_camera cv_camera`


## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Chris Ibrahim** 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

This is a non-exhaustive list of the people whose codes were used to achieve this project :

* Octavio Arriaga ([face classification package](https://github.com/oarriaga/face_classification))
* Francisco Lera (https://github.com/FranLera/simple_face_detection))

