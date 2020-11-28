from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import re


# Base ACM webpage
BASE_URL = "https://dl.acm.org"

# Author A
AUTHOR_A = BASE_URL + "/profile/81351593425/publications?AfterYear=1989&BeforeYear=2005&startPage=1&Role=author&pageSize=200"

# Author B
AUTHOR_B = BASE_URL + "/profile/81350576309/publications?AfterYear=1977&BeforeYear=2005&Role=author&startPage=0&pageSize=350"

# List item containing publication link and summary data
SUMMARY_LI_CLASS = "search__item issue-item-container"

# Span class holding links to papers
TITLE_SPAN_CLASS = "hlFld-Title"

# Span class containing publishing date
DATE_SPAN_CLASS = "dot-separator"

# Truncated abstract in summary data
ABSTRACT_TRUNC_DIV_CLASS = "issue-item__abstract"

# DIV class containing full abstract
ABSTRACT_DIV_CLASS = "abstractSection abstractInFull"


def get_js_soup(url, driver):
    """
    Get dynamically loaded web content using webdriver - taken from sample code

    Parameters
    ----------
    url : string
        valid URL to scrape
    driver : object
        valid selenium web driver object

    Returns
    -------
    object
        BeautifulSoup object containing web page content
    """
    driver.get(url)
    res_html = driver.execute_script("return document.body.innerHTML")
    return BeautifulSoup(res_html, "html.parser")


def scrape_status(url):
    """
    Utility function to print URL currently being scraped

    Parameters
    ----------
    url : string
        URL being scraped
    """
    print("-"*20, f" Scraping {url} ", "-"*20)


def scrape_pubs(url, driver):
    """
    Scrape the publications for a specific author's URL

    Parameters
    ----------
    url : string
        URL of publications to scrape
    driver : object
        valid selenium web driver object

    Returns
    -------
    list
        list of publications
    """
    scrape_status(url)
    soup = get_js_soup(url, driver)

    data = []
    idx = 0

    # Loop over all list items containing summary publication data
    for summary in soup.find_all("li", class_=SUMMARY_LI_CLASS):
        abstract_trunc = summary.find("div", class_=ABSTRACT_TRUNC_DIV_CLASS).find("p")
        if abstract_trunc:
            anchor = summary.find("span", class_=TITLE_SPAN_CLASS).find("a")
            url = anchor["href"]
            title = anchor.string
            date_span = summary.find("span", class_=DATE_SPAN_CLASS)

            if date_span:
                date = date_span.find("span").string
                month = ""
                year = ""
                m = re.search(r"^(\w+)\s(\d\d\d\d),", date)
                if m:
                    month = m.group(1)
                    year = m.group(2)

                idx += 1
                data.append({
                    "index": idx,
                    "title": title,
                    "month": month,
                    "year": year,
                    "url": BASE_URL + url,
                })

    print("-"*20, f" Found {len(data)} publication urls with abstracts ", "-"*20)

    return data


def scrape_abstract(url, driver):
    """
    Scrape the full abstract from the publication URL

    Parameters
    ----------
    url : string
        publication URL to scrape
    driver : object
        valid selenium web driver object

    Returns
    -------
    string
        scraped abstract
    """
    scrape_status(url)
    soup = get_js_soup(url, driver)

    abstract = ""
    abstract_div = soup.find("div", class_=ABSTRACT_DIV_CLASS)
    if abstract_div:
        abstract = abstract_div.find("p").getText().replace("\n", "  ")
    else:
        print("CANT FIND ABSTRACT")

    return abstract


if __name__ == "__main__":
    # Setup Chrome web driver with headless browsing for silent browser launch
    # Also, set Chrome log level to 3 to remove console INFO messages
    options = Options()
    options.headless = True
    options.add_argument("--log-level=3")
    driver = webdriver.Chrome("./chromedriver", options=options)

    author = "B"
    if author == "A":
        author_url = AUTHOR_A
    elif author == "B":
        author_url = AUTHOR_B
    else:
        raise(f"Could not find info for author {author}")

    # Scrape the web page for all publications
    pub_data = scrape_pubs(author_url, driver)

    # Scrape each individual faculty web page for biographical info
    abstracts_pre2000 = []
    abstracts_post2000 = []
    abstracts = []
    for datum in pub_data:
        abstract = scrape_abstract(datum['url'], driver)
        abstracts.append({"index": datum['index'], "content": abstract})
        if int(datum['year']) < 2000:
            abstracts_pre2000.append({"index": datum['index'], "content": abstract})
        else:
            abstracts_post2000.append({"index": datum['index'], "content": abstract})

    print("-"*20, f" Found {len(abstracts_pre2000)} pre-2000 abstracts ", "-"*20)
    print("-"*20, f" Found {len(abstracts_post2000)} post-2000 abstracts ", "-"*20)

    # Close the driver when finished scraping
    driver.close()

    # Write the publication metadata to file
    with open(f"author_{author}_meta.csv", "w") as f:
        f.write("ID,Title,Month,Year,URL\n")
        for datum in pub_data:
            f.write(f"{datum['index']},\"{datum['title']}\",{datum['month']},{datum['year']},{datum['url']}\n")

    # Write the abstracts to file:
    with open(f"author_{author}_abstracts.csv", "w", encoding="utf-8") as f:
        f.write("ID,Abstract\n")
        for abstract in abstracts:
            f.write(f"{abstract['index']},\"{abstract['content']}\"\n")

    with open(f"author_{author}_abstracts_pre2000.csv", "w", encoding="utf-8") as f:
        f.write("ID,Abstract\n")
        for abstract in abstracts_pre2000:
            f.write(f"{abstract['index']},\"{abstract['content']}\"\n")

    with open(f"author_{author}_abstracts_post2000.csv", "w", encoding="utf-8") as f:
        f.write("ID,Abstract\n")
        for abstract in abstracts_post2000:
            f.write(f"{abstract['index']},\"{abstract['content']}\"\n")