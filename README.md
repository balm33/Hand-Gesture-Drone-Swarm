Included are all the files that I used to control the group of drones via hand tracking
My pre-trained models are included in this repository. To train other models you must populate a folder called HandImages in this directory, with subfolders named after the intended gestures, and then populate those with their respective images. Then, train_gestures.py and train_gestures_nn.py can be used to train on that data.


First, the server needs to be started via
> python3 swarm_server.py

Next, in a seperate terminal window run the gesture recognition program
> python3 hand_tracking.py

Finally, you can start an instance in Webots, and the drones should connect to the server and begin receiving commands.
