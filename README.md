# Intro

This is our final project for UMich EECS 595 Adv. NLP Fall 2022. 

# Run on local machine

Load a local dataset and push it to hub:
```
python3 push_dataset_to_hub.py
```
Train the model on a datatset:

```
python3 model_train.py
```

Evaluate the model on a dataset for task 1 (given an image, rank all the actions)
```
python3 model_evaluate.py
```


Evaluate the model on a dataset for task 2 (given an action, rank all the candidate images)
```
python3 model_evaluate_task2.py
```

# Run on slurm server
submit the task
```
sbatch task1.sh
```
View jobs status
```
squeue -u <username>
squeue -A <accountname>
```
