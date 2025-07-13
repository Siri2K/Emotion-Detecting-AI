# Emotion Detecting AI

## Group AK-16

### Names
- Theebika Thiyagaraja Iyer - Theebika6 - 40191001
- Sriram Kanagaligham - Siri2K - 40170212
- Vithujanan Vigneswaran - Houdini29 - 40157822

### Roles
- Data Specialist : Theebika Thiyagaraja Iyer
- Training Specialist : Sriram Kanagaligham
- Evaluation Specialist : Vithujanan Vigneswaran

## Project
The project has 2 folders of interest: resource and src. 

Resources folder holds images files used for training and testing. Both Images are divided into classes (folders)
based on the emotion:
- Angry
- Focused
- Happy
- Neutral

src holds the code to perform the data cleaning and data manipulation using data.py. The main.py files is used to 
execute the entire program.

### Program Execution Steps
1. Open a Terminal and activate an anaconda environment

2. Clone repository

3. Navigate to the cloned project. Make sure that the current is at the 472 project. Path : 
    > ...\472_project

4. Download the necessary libraries using the command below:
    > conda install --file libraries.txt

5. Enter the following commands to perform the following:
   - Visualize the dataset (On Linux or Mac)
        > python src/main.py --display
   - Visualize the dataset (On Windows)
        > python src\main.py --display

   - Train and Test CNN Model (On Linux or Mac)
        >    python src/main.py --base
   - Train and Test CNN Model (Window)
        >    python src\main.py --base

   - Train and Test CNN Variant 1 (On Linux or Mac)
        >    python src/main.py --variant1
   - Train and Test CNN Variant 1 (Window)
        >    python src\main.py --variant1

   - Train and Test All CNN (On Linux or Mac)
        >    python src/main.py --trainAll
   - Train and Test All CNN (Window)
        >    python src\main.py --trainAll

   - Train and Test CNN Variant 2 (On Linux or Mac)
        >    python src/main.py
   - Train and Test CNN Variant 2 (Window)
        >    python src\main.py
