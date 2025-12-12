# DV - PHYSICAL ALBUM SALES
import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

URL = "https://soridata.com/physical_sales.html?rank=sales&gto=1&gendero=0"

def scrape_physical_sales_colab(url=URL, chrome_path="/usr/bin/google-chrome-stable"):
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")

    chrome_options.binary_location = chrome_path

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )

    try:
        driver.get(url)
        time.sleep(5)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        table = soup.find("table")

        if table is None:
            raise ValueError("Could not find any table on the page.")

        df_list = pd.read_html(str(table))
        if not df_list:
            raise ValueError("pandas.read_html returned no tables.")

        df = df_list[0]

        df = df.rename(columns={
            df.columns[0]: "rank",
            df.columns[1]: "artist",
            df.columns[2]: "sales"
        })

        df["sales"] = (
            df["sales"]
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.extract(r"(\d+)", expand=False)
            .astype(int)
        )

        return df

    finally:
        driver.quit()

df_sales = scrape_physical_sales_colab()
print(df_sales.head(20))

# Save inside GitHub repo folder: data/raw/
df_sales.to_csv("data/raw/kpop_physical_sales.csv", index=False, encoding="utf-8-sig")

print("Saved to data/raw/kpop_physical_sales.csv")

