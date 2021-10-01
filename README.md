# Hand_gesture_recognition_CNN
Gesture Recognition using CNN to control your device.

Please download anaconda  64 bit software 

And then open the anaconda command prompt in administrator mode and type the following command

conda create -n project python=3.7

activate project 

pip install tensorflow==1.14 keras==2.1.2 seaborn imutils sklearn opencv-python==3.4.2.16

pip install keyboard

pip install h5py==2.10.0(For loading already existed h5py file).



STEP 1
You can create your own dataset by using detect.py file in Create_Dataset folder.Change the folder_name in detect.py line 10 and collect data for multiple classes.Do not collect gestures before the background is calibrated for which i have used timer of 230ms on line 73 (can change it according to your device processing speed ).

STEP 2
Run the Dataset prepration file with command python Dataset_Prep.py -i [images_folder_path].
Run this file twice by commenting the line 33 and uncommenting the line 34.Take note that you have to only run first [cv2.imwrite(imagePath,crop)] than run  [cv2.imwrite(imagePath,crop_resized)] so that the image is first cropped than it is resized in 100*100 pixels.

STEP 3
Open anaconda and run file simple_cnn.py by copying its code into ipynb file for simple dataset training.
Or You can train the model with bottleneck mothod where you have to provide training data,validation data and test data 
The trained model will be saved in model.h5 file.

STEP 4(can be directly used as model.h5 file already exist)
To run the final application run Application_Key_press which will load the model and you can control your device.


For more details about the project,click the link below
https://turcomat.org/index.php/turkbilmat/article/view/6811

Tutorial Video Link below-
https://youtu.be/lvGQ214P4RA





