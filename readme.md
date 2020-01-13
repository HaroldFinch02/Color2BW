# Encoder-Decoder network to convert Color images to Black&White

### Instructions to run:

* Make sure you have all required setup for tensorflow GPU(step available on tensorflow website)
* Install the requirements using pip(I recommend a virtual environment)
* Run: *python3 testing.py*
* Enter name of image to be converted
* The converted image will be saved as res.jpg folder

### Retraining the model
* Copy any color images dataset to train_data/originalImages folder(I had used animals dataset from kaggle - not uploaded to github)
* Run: *python3 data/createData.py*
    * This will resize the images to 128x128 and save both color and grayscale images in respective folders
* Repeat the same for test images (Did not write that in script :sweat_smile: )
* Change the number of training and testing image variables as per your dataset in training.py file
* Run: *python3 training.py* (You can modify the epochs and other parameters if you like)
* Model will be saved in SavedModel folder

### Note
I wrote this code because i wanted to try out encoder-decoder type of models. There must be a lot of better ways/models to do this. 