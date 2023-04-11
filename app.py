from datetime import datetime, timedelta
import os
import time
from langchain import OpenAI
from langchain.docstore import document
from langchain.chains.summarize import load_summarize_chain
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

    for post in ai_posts:
        try:
            if post["url"] is not None:
                logger.info(f"{post['url']} with {post['points']} points and {post['num_comments']} comments")
                #get_story_summary(post["url"])
        except:
            logger.error("Error getting story summary for post {}", post["url"])
        time.sleep(1)

def get_story_summary(story_url):
    # we need to use a custom prompt to ask to discard summaries when the message refers to not having Javascript or cookies enabled
    # the custom prompt should also disregard content not related AI

    BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")
    loader = BeautifulSoupWebReader()
    print(f"Getting summary for {story_url}...")
    documents = loader.load_data(urls=[story_url])

    chain = load_summarize_chain(OpenAI(temperature=0.7), chain_type="map_reduce")

    # create a langchain document for each text chunk
    docs = [document.Document(page_content=documents[0].text[:4096])]
    print(chain({"input_documents": docs}, return_only_outputs=True))


if __name__ == "__main__":
    get_ai_posts()
