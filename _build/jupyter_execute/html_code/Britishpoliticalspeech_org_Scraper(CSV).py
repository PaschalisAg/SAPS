#!/usr/bin/env python
# coding: utf-8

# # Basic Britishpoliticalspeech.org Scraper (CSV)
# 
# This python based scraper will scrape British political speeches from political leaders in the UK from [Britishpoliticalspeech.org](http://britishpoliticalspeech.org/). When fully run the scraper will output a CSV file containing basic metadata about the speeches and the speeches themselves. These could for further analysis with for instance tools from the Pandas library.

# In[2]:


import sys
import csv
import requests
import re
from bs4 import BeautifulSoup


# In[3]:


# This function loads a webpage
def load_page(url):
    with requests.get(url) as f:
        page = f.text
    return page


# ## Locate the Data
# 
# Here we define two functions. First we extract metadata from the [main content table](http://britishpoliticalspeech.org/speech-archive.htm) of the archive using `get_speech_data()`. Second we define a function to look at specific speech pages linked in the content table using the `get_speech()` function. 
# 
# From the main content table we extract data on:
# - name of the speech
# - date on which the speech was held
# - party to which the speaker of the speech belonged
# - the hyperlink to the specific speech page
# 
# Additionally an id is added to every speech. 

# In[4]:


def get_speech_data(url):
    content_page = BeautifulSoup(load_page(url), 'lxml')       #Open the webpage
    if not content_page:                                            
        print('Something went wrong!', file=sys.stderr)
        sys.exit()
    data = []
    for count, row in enumerate(content_page.find_all('tr')[2:]): #Find the data we are looking for
        dates = row.find_all('td')[0]
        parties = row.find_all('td')[1]
        speakers = row.find_all('td')[2]
        speech = row.find_all('td')[3]
        link = row.find('a').get('href')
        data.append({                               #Add the data to a dictionary
            'id' : parties.text + '_' + str(count),
            'date': dates.text,
            'party': parties.text,
            'name speech': speech.text,
            'link': 'http://britishpoliticalspeech.org/' + link
        })
    return data 


# From the specific speech page we extract data on:
# - the full speech text 
# - the name of the speaker (collected here as it was incomplete in the main content table list)
# - the location in which the speech was held (easier to scrape here)
# 
# In this function we skip speeches in which the speech text is not available due to copyright. 

# In[5]:


def get_speech(url):
    speech_page = BeautifulSoup(load_page(url), 'lxml')                  #Open the webpage
    interesting_html = (speech_page.find(class_='speech-content').text.strip()
        .replace('\xa0\n', '').replace('\n','').replace('\x85','').replace('\u2011','')) #Find the full text of the speech
    skip_check = 'Owing to a copyright issue this speech has been removed.' #Check of this text is in the speech, otherwise this can be skipped
    speaker_html = speech_page.find(class_='speech-speaker').text.strip().split('(', 1)[0] #Find the speaker of the speech
    location_html = speech_page.find(class_='speech-location').text.strip() #Find the location at which the speech was held
    if 'Location: ' in location_html:
        location_html = location_html.replace('Location: ', '')
    if not interesting_html or skip_check in interesting_html: # or not speaker_html or not location_html don't really care about not finding these
        #print('Skipped - No information available for {}'.format(url), file=sys.stderr)
        return {}                                                      
    return {'speech' : interesting_html, 'speaker' : speaker_html, 'location' : location_html} #Add the data to a dictionary


# ## Scraping the Data
# 
# The following code will proceed to apply the previously made functions for scraping the desired data and writes the output in a csv file.

# In[6]:


#This code applies the scraping functions
index_url = 'http://britishpoliticalspeech.org/speech-archive.htm'         # Contains the list of speeches
speech_data = get_speech_data(index_url)                      # Get speech metadata
for row in speech_data:
    #print('Scraping info on {}.'.format(row['name speech'])) # Might be useful for debugging
    url = row['link']
    speech_info = get_speech(url)                    # Gets the speeches themselves
    for key, value in speech_info.items():
        row[key] = value                              # Add the new data to our dictionary
print('Done scraping!')


# In[7]:


#This code writes the data in the dictionary in a csv file
with open('metadata.csv', 'w', encoding='utf-8') as f:       # Open a csv file for writing
    fieldnames=['id','speaker', 'party', 'location', 'date', 'name speech',
                'speech']                                 # These are the values we want to store
    writer = csv.DictWriter(f,
                            delimiter=',',                # Common delimiter
                            quotechar='"',                # Common quote character
                            quoting=csv.QUOTE_NONNUMERIC, # Make sure that all strings are quoted
                            fieldnames=fieldnames
                            )
    writer.writeheader()                                  # Create headers in our csv file
    for row in speech_data:
        writer.writerow({k:v for k,v in row.items() if k in fieldnames})


# ## Reading the Metadata
# 
# In this last part you can run the following code to make a tabular overview (with pandas) of the data stored in the metadata csv file and check if the metadata has been properly scraped. 

# In[8]:


import pandas as pd

df = pd.read_csv('metadata.csv')
df


# In[ ]:




