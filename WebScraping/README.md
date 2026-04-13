# 🌐 Web Scraping using Python (Flipkart Laptops Data)

## 🔍 Overview

This project uses web scraping techniques to extract laptop product details (name and price) from Flipkart using Python.

## 🎯 Objective

To automate data extraction from e-commerce websites and store the scraped data for analysis.

## 📊 Data Source

* Website: Flipkart (Laptops category)
* Extracted Data:

  * Product Names
  * Prices

## 🛠 Tools & Technologies

* Python
* Selenium (for browser automation)
* BeautifulSoup (for HTML parsing)
* Pandas (for data storage)
* OpenPyXL (for Excel export)

## ⚙️ Methodology

### 📌 Steps Performed:

1. Opened Flipkart website using Selenium WebDriver
2. Extracted page source (HTML)
3. Parsed HTML using BeautifulSoup
4. Identified product elements using class tags
5. Extracted:

   * Laptop names
   * Prices
6. Stored data into a structured format using Pandas
7. Exported final dataset to Excel file

## 📈 Output

* Extracted product and pricing data
* Stored in Excel file (`flaptops.xlsx`)

## 📁 Files Included

* Python script (web scraping code)
* Output Excel file with scraped data

## 🔑 Key Learnings

* Web scraping using Selenium and BeautifulSoup
* Handling dynamic websites
* Data extraction and storage
* Automating repetitive data collection tasks

## ⚠️ Note

* Website structure may change, which can affect scraping results
* Scraping should follow website terms and conditions

## 📌 Conclusion

This project demonstrates how web scraping can be used to collect real-time data from websites for analysis and decision-making.

## 📫 Author

Rahul Bhujade
