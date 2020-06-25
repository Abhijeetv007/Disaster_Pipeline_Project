# Disaster Response Pipeline Project

## Project Motivation:

I made use of the data engineering concepts to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.
## File Structure:

	- app
	 -- template
	 	- master.html  # main page of web app
	 	- go.html  # classification result page of web app
	 -- run.py  # Flask file that runs app

	- data
	- disaster_categories.csv  # data to process 
	- disaster_messages.csv  # data to process
	- process_data.py
	- InsertDatabaseName.db   # database to save clean data to

	- models
	- train_classifier.py
	- classifier.pkl  # saved model 

	- README.md

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Example 

Type a message :Hurricane lead to water and food shortage in costa rica

### Commit
1-Change some spelling in app's code
2-Explaning the code
3-
