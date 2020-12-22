from bs4 import BeautifulSoup as bs
import pandas as pd
import requests

cities = ["Houston", "Austin", "Dallas", "San-Antonio"]
data = []

for city in cities:
    for i in range(1, 50):
        if(i == 1):
            resp = requests.get("https://homefinder.com/for-sale/TX/Houston")
        else:
            resp = requests.get("https://homefinder.com/for-sale/TX/Houston?page={}".format(i))
        soup = bs(resp.text, "html.parser")
        prices = [i.find('h4').text.replace("$", "").split() for i in soup.find_all("div", {"class":"attributes"})]
        attributes = [i.text.split("|") for i in soup.find_all("div", {"class":"text-muted"})]
        attributes = [[j.strip() for j in i] for i in attributes]
        data += [i[:2] + j for i,j in zip(prices,attributes) if "Sale" in i and len(i) == 4]

df = pd.DataFrame(data)
df.columns = ["Price", "Type", "Bedrooms", "Bathrooms", "Square Feet"]
df.dropna(inplace=True)
df[["Bedrooms", "Bathrooms", "Square Feet"]] = df[["Bedrooms", "Bathrooms", "Square Feet"]].applymap(lambda x: int(x.split()[0].replace(",", "")))
df["Price"] = df["Price"].apply(lambda x: int(x.replace(",", "")))
df.to_csv("house_prices.csv", index=False)