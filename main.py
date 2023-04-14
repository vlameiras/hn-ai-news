from datetime import datetime, timedelta
import os
import sqlite3
import time
from jinja2 import Environment, FileSystemLoader
from langchain import OpenAI, PromptTemplate
from langchain.docstore import document
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from llama_index import download_loader
from loguru import logger
import openai
import requests


# Set up OpenAI authentication
openai.api_key = os.environ.get("OPENAI_API_KEY")

ai_keywords = {
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "neural networks",
    "natural language processing",
    "computer vision",
    "openai",
    "gpt-3",
    "gpt-4",
    "ai",
    "ml",
    "nlp",
    "speech recognition",
    "stable diffusion",
    "transformers",
    "bert",
    "gpt",
    "gpt-2",
    "dall-e",
    "midjourney",
    "chatbot",
    "chatgpt",
    "copilot",
    "llama",
    "llama index",
    "lora",
    "gpt4all"
}

def get_ai_posts():
    api_url = "https://hn.algolia.com/api/v1/search"
    yesterday = datetime.now() - timedelta(days=1)
    numeric_filters = f"created_at_i>{int(yesterday.timestamp())}"
    ai_posts = []
    
    for keyword in ai_keywords:
        print(f"Searching for posts related to '{keyword}'...")
        query = f'"{keyword}"'
        api_params = {
            "query": query,
            "numericFilters": numeric_filters,
        }
        response = requests.get(api_url, params=api_params)
        response.raise_for_status()
        data = response.json()

        # add posts to ai_posts list if post["url"] does not exist in a post already in ai_posts
        for post in data["hits"]:
            if post["url"] not in [p["url"] for p in ai_posts]:
                ai_posts.append(post)

        # sort ai_posts by number of comments and points
        # discard num_comments and points if they are None
        ai_posts.sort(key=lambda x: (x["num_comments"] or 0) + (x["points"] or 0), reverse=True)

    logger.info(f"Found {len(ai_posts)} posts related to AI")

    # store the summaries in a sqlite database indexed by the post url
    # only get the summary if the post url is not already in the database
    new_ai_posts = []

    # Set up sqlite3 database
    conn = sqlite3.connect("ai_posts.db")
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS posts
                   (url TEXT PRIMARY KEY, summary TEXT)''')

    for idx, post in enumerate(ai_posts):
        try:
            if post["url"] is not None:
                # Check if post URL is already in the database
                cur.execute("SELECT url, summary FROM posts WHERE url=?", (post["url"],))
                url_in_db = cur.fetchone()

                if not url_in_db:
                    if idx > 5:
                        post["summary"] = get_story_summary_2(post["url"], short=True)
                    else:
                        post["summary"] = get_story_summary_2(post["url"], short=False)

                    cur.execute("INSERT INTO posts (url, summary) VALUES (?, ?)", (post["url"], post["summary"]))
                    conn.commit()
                else:
                    post["summary"] = url_in_db[1]

                new_ai_posts.append(post)

        except:
            logger.error("Error getting story summary for post {}", post["url"])
        time.sleep(0.1)


    conn.close()
    #generate_email(new_ai_posts)
    generate_static_text(new_ai_posts)

def get_story_summary_2(story_url, short=False):
    BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")
    loader = BeautifulSoupWebReader()
    print(f"Getting summary for {story_url}...")
    documents = loader.load_data(urls=[story_url])

    llm = OpenAI(temperature=0.7)

    if short:
        prompt_template = """The original text was scraped from a website, so first you need to check if the text is just a response stating that Javascript or cookies must be enabled. If it is, the output of your message should be "N/A". Also check if the text is related to AI, if it isn't write the ouput "N/AB" . Write a concise one line summary of the following and include an emoji at the beginning of the summary which portrays the text sentiment:


        {text}

        """
    else:
        prompt_template = """The original text was scraped from a website, so first you need to check if the text is just a response stating that Javascript or cookies must be enabled. If it is, the output of your message should be "N/A". Also check if the text is related to AI, if it isn't write the ouput "N/AB" . Write a concise summary of the following and include an emoji at the beginning of the summary which portrays the text sentiment:


        {text}

        """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)

    docs = [document.Document(page_content=documents[0].text[:4096])]
    return chain.run(input_documents=docs, lang='english', return_only_outputs=True)
    
def generate_email(posts):
    # Configure Jinja2 to load templates from the current directory
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('email-template.jinja2')

    # convert posts to an articles array
    articles = []
    for post in posts:
        article = {
            "title": post["title"],
            "summary": post["summary"],
            "link": post["url"],
            "hn_discussion_link": f"https://news.ycombinator.com/item?id={post['objectID']}"
        }
        articles.append(article)

    rendered_html = template.render(articles=articles)

    filename = f"rendered-{datetime.now().strftime('%Y-%m-%d')}.html"
    with open(filename, "w") as file:
        file.write(rendered_html)

    print(f"Rendered HTML saved to {filename}")

def generate_static_text(posts):
    """Iterate the list of posts and print the title, URL, HN discussion link, and summary
       Separate posts with a line of dashes
       Check if the post fields exist to prevent KeyError. If it doesn't add a dash
       Write the output to a text file
    """
    for post in posts:
        print(post["title"])
        print(post["url"])
        print(f"https://news.ycombinator.com/item?id={post['objectID']}")
        print(post["summary"])

        #write to file named with the current date
        filename = f"rendered-{datetime.now().strftime('%Y-%m-%d')}.txt"
        with open(filename, "a") as file:
            # check if post has a summary
            if post.get("summary"):
                file.write(post["title"] + "\n " + post["summary"] + "\n" + post["url"] + "\n" + f"https://news.ycombinator.com/item?id={post['objectID']}"+ "\n\n")
            else:
                file.write(post["title"] + "\n " + "-" + "\n" + post["url"] + "\n" + f"https://news.ycombinator.com/item?id={post['objectID']}"+ "\n\n")
        
        print("-" * 1)

if __name__ == "__main__":
    get_ai_posts()
