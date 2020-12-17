# Statistical Learning for Engineers -- Final Project

This is our submission for the Final Project. Adyota worked on Part 1, and Adithya worked on Part 2. The assembled PDF report is in the repo as well for reference.



## Part 1 -- Machine Learning
Data, processed data, and trained model parameters are available at https://mega.nz/folder/NtBgXLyZ#-97kmN4b4AjvZFf5092M-g

To run the first project, please download the fruit_dataset.zip file and unzip the file. Place the folder in the same directory as the python code. Furthermore, change the SAVE_PATH accordingly. For the models that I obtained when running, they can be downloaded and loaded using model.load. Also the X and y after data processing have been saved as numpy objects and can be loaded as well. The history variables for the four ANN models are pickle objects and can be trivially loaded. The ouputs from the training and evaluating of each model is located in the output.txt file. Finally, the conda environment yml file is attached, so it can be run in the same environment I used.


To run,
```
conda activate SL_Final
python Project_P1.py
```


One word of warning: I used a Quadro RTX 8000 to run my code, which uses 48 GB of memory. For the Polynomial and Logistic regression models, I load the ENTIRE dataset into the GPU. I'm not entirely sure how much memory in total was used by torch, since it reserved the entire 48 GB, but keep in mind that if "Out of Memory" Errors occur, a DataLoader object may need to be used.



## Part 2 -- Reinforcement Learning
Code for the reinforcement Learning part is available as two jupyter notebooks:

Project_P2_DP.ipynb contains the code that performs dynamic programming. This can likely be run as is because it only requires numpy and matplotlib. I have imported gym as well into this notebook but it is not required. The code should therefore be able to run as is. 

Project_P2_QL.ipynb contains the code that performs Q-learning. This requires the installing and importing of openai gym. To do this, if one is using anaconda navigator, first a channel that contains the gym package needs to be added. This can be done clikcing on the channels button on the Navigator home screen. Then the following URL is to be added : https://conda.anaconda.org/conda-forge. Next the "gym" package can be searched for in the "Environments" section of the Navigator window and installed. Once done gym can be imported. Alternatively one can just import the enivronment file "QL_environment.yml" in the Project2 folder. 
NOTE : As I am running 20000 episodes of gym the code takes a little while to run. This can be reduced to 2000 to save on time.
