from langchain.embeddings.base import Embeddings
from typing import List
import requests
import asyncio
import aiohttp

class LocalLlamaEmbeddings(Embeddings):
    def __init__(self, url: str, headers: dict):
        self.url = url
        self.headers = headers
        super().__init__()

    def api_query(self, url: str, headers: dict, json: dict):
        response = requests.post(url=self.url, headers=self.headers, json=json)
        response = response.json()
        return response["data"][0]["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        async def fetch_data(session, url, payload):
            async with session.post(url, json=payload) as resp:
                data = await resp.json()
                return data["data"][0]["embedding"]

        async def post_multiple():
            async with aiohttp.ClientSession() as session:
                tasks = []
                for text in texts:  # replace with your range
                    url = self.url  # replace with your API endpoint
                    payload = {"input": text}  # replace with your payload
                    tasks.append(fetch_data(session, url, payload))  # Append the coroutine

                responses = await asyncio.gather(*tasks)

            return responses

        # Call the async function using asyncio.run()
        embeddings = asyncio.run(post_multiple())
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.api_query(url=self.url, headers=self.headers, json={"input": text})
