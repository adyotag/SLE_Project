# Statistical Learning for Engineers -- Final Project


# Part 1 -- Machine Learning
Data, alongside model is available at https://mega.nz/folder/NtBgXLyZ#-97kmN4b4AjvZFf5092M-g

To run the first project, please download the fruit_dataset.zip file and unzip the file. Place the folder in the same directory as the python code. Furthermore, change the SAVE_PATH accordingly. For the models that I obtained when running, they can be downloaded and loaded using model.load. Also the X and y after data processing have been saved as numpy objects and can be loaded as well. The history variables for the four ANN models are pickle objects and can be trivially loaded. The ouputs from the training and evaluating of each model is located in the output.txt file. Finally, the conda environment yml file is attached, so it can be run in the same environment I used.


To run, 
{
conda activate SL_Final
python Project_P1.py
}


One word of warning: I used a Quadro RTX 8000 to run my code, which uses 48 GB of memory. For the Polynomial and Logistic regression models, I load the ENTIRE dataset into the GPU. I'm not entirely sure how much memory in total was used by torch, since it reserved the entire 48 GB, but keep in mind that if "Out of Memory" Errors occur, a DataLoader object may need to be used.



# Part 2 -- Reinforcement Learning
