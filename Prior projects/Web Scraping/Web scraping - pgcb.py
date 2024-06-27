# ---------- Import libraries
import requests
import urllib.request
import time
from bs4 import BeautifulSoup

# Bangla to English
from googletrans import Translator
translator = Translator()
import pandas as pd
from datetime import date
from datetime import datetime

# datetime_object = datetime.strptime('01 Jun 2005', '%d %m %Y')
# ---------- Extract the dataset
data_rows = []
for i in range(1, 920): #926 920
    url = 'https://web.pgcb.gov.bd/imported_power_bn?page=' + str(i)
    response = requests.get(url)
    # find the whole content from that link
    soup = BeautifulSoup(response.text, "html.parser")
    # print(soup.prettify())
    # extract only that table with the data
    data = soup.find('tbody')
    for tr in data.find_all("tr"):
        cells = []
        # grab all td tags in this table row
        tds = tr.find_all("td")
        for td in tds[0]:
            cells.append(translator.translate(td.strip(), src='bn', dest='en').text)
        for td in tds[1]:
            cells.append(translator.translate(td.strip(), src='bn', dest='en').text)
        for td in tds[2]:
            cells.append(int(td.strip()))
        for td in tds[3]:
            cells.append(int(td.strip()))
        for td in tds[4]:
            cells.append(int(td.strip()))
        for td in tds[5]:
            cells.append(int(td.strip()))
        data_rows.append(cells)

    data_rows  # data_raw needs to be appended

# save the dataset
# df_pgcb = pd.DataFrame(data_rows)
# data_rows.dtypes
# Find the columns' names
# list(data_rows.columns)
# Change the columns' format from text to date/time
# data_rows[0] = pd.to_datetime(data_rows[0], format='%dd-%mm-%YYYY')
# date_time_obj = datetime.strptime(a, '%d-%m-%y')
# date_time_obj = datetime.strptime(data_rows[0], '%d-%m-%y')

# ----------- Convert Bengali to English
# cells.append(translator.translate(td.strip(), src='bn', dest='en').text)



# Create the CSV file
df_pgcb_file = pd.DataFrame(data_rows).to_csv("pgcb_dataset_page11.csv")
