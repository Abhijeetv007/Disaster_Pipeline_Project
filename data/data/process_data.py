# import libraries
from sqlalchemy import create_engine
import sys
import pandas as pd

def load_data(messages_filepath, categories_filepath):
    """
    This function loads the message and categories files and
    merge them and return the new dataframe for the project
    """
    # Read messages and categories data
    messaging = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge the two dataframes
    dataframe = messaging.merge(categories, how='inner', on= 'id')
    return dataframe
    


def clean_data(dataframe):
    """
        Cleaning the merged dataframe to make it ready to analyze
    """
    # split categories into seperate
    categories = dataframe.categories.str.split(';', expand=True)
    
    # select the first row&col of the categories dataframe
    row&col = categories.iloc[0]
    cate_col = row&col.apply(lambda x: x[:-2])
    cate.columns = cate_colnames
    
    #convert categories values to numeric instead of strings
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
    
    # replace categories column in dataframe 
    dataframe.drop(columns = ['categories'], inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    dataframe = dataframe.join(categories)
    
    #drop duplicates
    dataframe.drop_duplicates(inplace=True)
    
    return dataframe

def save_data(dataframe, database_filename):
    """
   Take the input dataframe and save it into sqlite database
    """
    # Creating sqlite engine and save the dataframe with the name message
    engine_process = create_engine('sqlite:///'+ database_filename)
    dataframe.to_sql('messaging', engine_process, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        dataframe = load_data(messages_filepath, categories_filepath)
        print('Cleaning data...')
        dataframe = clean_data(dataframe)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(dataframe, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
