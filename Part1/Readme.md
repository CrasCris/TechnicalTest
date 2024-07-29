# technical_test_DS_GenAI - Part1

This section has the code for the firts part of the technical test.

The Jupyter Notebook have Traditional ML skills, also include the main.py the backend api using fastapi framework for production and the dockerfile to build and run it, in windows or linux OS.

The MachineLearningSkills.ipynb has the steps of our model and the main.py is out endpoint to use in production.

### Futere improve

* Use a bigger param_grid to improve the finetuning
* Create a structure Frontend to our final users
* Try with multiple models


## How to running in production with Docker


1. `cd Part1` 

2. `docker build -t part1:test .`

3. `docker run -d -p 8080:8080 --rm part1:test `