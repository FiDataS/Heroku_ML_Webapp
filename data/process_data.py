#Import neccessary librarys
import sys
import pandas as pd
from sqlalchemy import create_engine
from langdetect import detect
from collections import Counter

def load_data(messages_filepath, categories_filepath):
    """
    Loads data from csv files, drops dublicate-ids and
    merges the message and categories data into one datafram
    
    :param messages_filepath: filepath of messages csv file (string)
    :param categories_filepath: filepath of categories csv file (string)
    
    :returns df: dataframe of merged messages and categories data (dataframe)
    """
    #load messages dataset
    messages = pd.read_csv(messages_filepath)
    #load categories dataset
    categories = pd.read_csv(categories_filepath)
    #drop dublicates (depending on id) for both datasets before merging
    messages.drop_duplicates(inplace=True, subset='id')
    categories.drop_duplicates(inplace=True, subset='id')
    #merging into one df
    df = pd.merge(messages, categories, on=['id'])
    return df

def clean_data(df):
    """
    Cleans the df of the loaded data and returns a new dataframe with
    separate columns per classification/category with binary values
    
    :param df: dataframe of loaded data (dataframe)
    
    :returns df_n: clean dataframe (dataframe)
    """
    #split the categories in df into 36 individual category columns
    categories = df.categories.str.split(pat=';', expand=True)
    #get column names for the categories and use the values up to the second last character of each string
    row=categories.iloc[0]
    category_colnames = list(row.apply(lambda c: c[0:-2]))
    #rename the columns of categories
    categories.columns = category_colnames
    #convert category values to just number 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda v: v[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    #replace categories column in df with the new category columns
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df_n = pd.concat([df, categories], axis=1)
    #remove rows with unplausible values (non binary values - not 0 or 1 - in the categories
    for column in categories.columns:
        df_n = df_n[df_n[column] < 2]
    #remove rows that do not have any letters in their original message
    #Make sure every message is a string
    df_n['original'] = df_n['original'].astype('str')
    #Drop columns that do not have any letters in their original message
    #First get indices of rows where that is the case:
    indexNames = df_n[~df_n['original'].str.contains('[A-Za-z]')].index
    #then drop those rows with:
    df_n.drop(indexNames, inplace=True)
    #return the new dataframe
    return df_n

def country_count(df):
    """
    Creates a dataframe that shows where the original messages were from
    Result: Dataframe with Countries in Dataset and Counts
    
    :param df: cleaned dataframe (dataframe)
    :returns df_c: dataframe of countries and counts (dataframe)
    """
    #Detect the languages for each original message
    country_list = [detect(x) for x in df['original']]
    #Count these with Counter
    counter_countries=Counter(country_list)
    #Create dataframe
    df_c=pd.DataFrame(counter_countries.most_common(), columns=['country','count_messages'])
    return df_c
    
def save_data(df, df_c, database_filename):
    """
    Saves cleaned data in a database
    Saves both the training dataframe and the dataframe with countries and counts in different tables
    
    :param df: cleaned dataframe (dataframe)
    :param database_filename: filename of the database used for storing (string)
    """
    #create engine for database
    engine = create_engine('sqlite:///' + database_filename)
    #save the df into the sqlite database
    df.to_sql('Categorized_Messages', engine, if_exists='replace', index=False)
    #save the df_c into the sqlite database
    df_c.to_sql('Country_Count', engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Creating Country Count Dataframe...')
        df_c = country_count(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, df_c, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'Disaster_Response.db')


if __name__ == '__main__':
    main()