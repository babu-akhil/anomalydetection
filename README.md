# Unsupervised video anomaly detection

Exploring unsupervised and semi supervised methods in video anomaly detection.
Currently working on UCSD-Ped 1 Dataset.


Training the model:

Edit train.py and put your dataset path name instead of the default one.

Dataset should be in this format:

Parent folder: 

* subfolder 1
  * im001
  * im002... 
  * im00n
 
* subfolder 2...
 
* subfolder n...

Then run train.py


Testing:

Edit test.py: Change SINGLE_TEST_PATH to the folder containing your test video and then run test.py

