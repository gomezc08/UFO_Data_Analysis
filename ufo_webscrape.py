import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# importing the data from the internet...
url = "https://www.nuforc.org/webreports/ndxevent.html"
r = requests.get(url)
html = r.text
soup = BeautifulSoup(html, features="html.parser")

# Find all the links on the page
links = soup.find_all('a')

# Extract the URLs of the monthly reports
urls = []
for i in links:
    #print(i)
    url = i.get('href')
    #print(url)
    # all urls of interest has ndxe in it...
    if 'ndxe' in url:
        urls.append(url)
# now, urls holds all the links to each link  
# add all important features to their appropriate lists
date = []
city = []
state = []
shape = []
country = []
for i in urls:
    url2 = "https://www.nuforc.org/webreports/" + i
    r2 = requests.get(url2)
    html2 = r2.text
    soup2 = BeautifulSoup(html2, features="html.parser")
    # each row is inside a tr (wanna skip the header row), and each row is inside a td
    rows = soup2.find_all('tr')[1:]
    # j here is every link in the initial page like every timestamp.
    for j in rows:
        cells = j.find_all('td')
        d = cells[0].get_text()#.strip()  # Remove leading/trailing spaces
        c = cells[1].get_text()
        st = cells[2].get_text()
        co = cells[3].get_text()
        s = cells[4].get_text()
        date.append(d)
        city.append(c)
        state.append(st)
        country.append(co)
        shape.append(s)

# turn it into a dataframe

df = pd.DataFrame({"Date" : date, "City" : city, "State" : state, "Country" : country,"Shape of UFO" : shape})
df.replace('', np.nan, inplace=True)
df.dropna(subset=['State', "City", "Country", "Shape of UFO"], inplace=True)
df = df.drop(df[df['City'] == 'null'].index)
df = df.drop(df[df['State'] == 'null'].index)
df = df.drop(df[df['Country'] == 'null'].index)
df = df.drop(df[df['Shape of UFO'] == 'null'].index)

# Convert the date column to a datetime object
df["Date"] = pd.to_datetime(df["Date"], errors='coerce')


# Extract numerical features from the date column
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Hour"] = df["Date"].dt.hour
df["Minute"] = df["Date"].dt.minute
df.dropna(inplace=True)

# turn it into a csv file
df.to_csv('ufo_data2.csv', index=False)
