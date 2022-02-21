#!/usr/bin/env python
# coding: utf-8

# # Basic Britishpoliticalspeech.org Scraper (TXT)
# 
# This python based scraper will scrape British political speeches from political leaders in the UK from [Britishpoliticalspeech.org](http://britishpoliticalspeech.org/). When fully run the scraper will output a directory with txt files of all the individual speeches held. These could be used for specific textual analyses.

# In[2]:


import sys
import requests
import re
import os
from bs4 import BeautifulSoup


# In[3]:


# This function loads a webpage
def load_page(url):
    with requests.get(url) as f:
        page = f.text
    return page


# ## Locate the speeches
# Here we define two functions to first extract the hyperlinks from the speeches from the main content table of the archive using `get_speech_data()`, and secondly to download the speech texts on the specific speech pages linked in the content table using the `get_speech()` function.

# In[4]:


def get_speech_data(url):
    speech_page = BeautifulSoup(load_page(url), 'lxml')  # Open the webpage 
    if not speech_page:                                            
        print('Something went wrong!', file=sys.stderr)
        sys.exit()
    data = []
    for row in speech_page.find_all('tr')[2:]:
        speech = row.find_all('td')[3] #Find the name of the every speech
        link = row.find('a').get('href') #Find the hyperlink for every speech
        data.append({
            'link': 'http://britishpoliticalspeech.org/' + link
        }) #Store the hyperlinks in 'data'
    return data


# In[5]:


def get_speech(url):
    speech_page = BeautifulSoup(load_page(url), 'lxml')    #Open the speech webpage
    interesting_html = (speech_page.find(class_='speech-content').text.strip()
        .replace('\xa0\n', '').replace('\n','').replace('\x85','').replace('\u2011','')) #Find the full text of the speech
    skip_check = 'Owing to a copyright issue this speech has been removed.' #Check of this text is in the speech, otherwise this can be skipped
    if not interesting_html or skip_check in interesting_html: # or not speaker_html or not location_html don't really care about not finding these
        #print('Skipped - No information available for {}'.format(url), file=sys.stderr)
        return {}
    return {'speech' : interesting_html} #returns the full text of the speech


# ## Scraping the Data
# 
# The following code will proceed to apply the previously made functions for scraping the desired data and writes the output in txt files in a newly created directory called "speeches". This directory will be in the created wherever you stored this notebook.

# In[6]:


index_url = 'http://britishpoliticalspeech.org/speech-archive.htm'         # Contains a list of speeches
list_speech_data = get_speech_data(index_url)                      # Get speeches with metadata
list_rows_to_remove = []
#print (" - - - - - " + str(len(list_speech_data)))

for count, row in enumerate(list_speech_data):
    #print('Scraping info on {}.'.format(row['name speech'])) # Might be useful for debugging
    url = row['link']
    speech_info = get_speech(url)                    # Get the speech, if available
    if speech_info == {}:
        list_rows_to_remove.append(count)
    else:    
        for key, value in speech_info.items():
            row[key] = value                              # Add the new data to our dictionary
    #print('Scraped info on {}.'.format(row['name speech']) + '\t from {}.'.format(row['speakers']))

for d_elem in reversed(list_rows_to_remove): # Delete list rows in reverse to avoid errors
    #print("Speech missing - Deleted: " + str(d_elem))
    del list_speech_data[d_elem]

#print (" - - - - - " + str(len(list_speech_data)))
print('Done scraping!')


# In[7]:


path = "speeches/"
# Check whether the specified path exists or not
isExist = os.path.exists(path)

if not isExist:  
    # Create a new directory because it does not exist 
    os.makedirs(path)
    print("The new directory is created!")

# Write the speeches in txt files with the id as file name
number = 1
for row in list_speech_data:
    filename = f'political_speech_{number}.txt'
    #filename = row['id']
    #print(filename)
    file1 = open(path + filename,"w")
    number += 1
    file1.writelines(row['speech'])
    file1.close() #to change file access modes    


# In[ ]:




