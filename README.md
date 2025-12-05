# SpotiSearch

**Goal:** This project aims to build a semantic search engine for albums and songs that retrieves albums and tracks based on their vibe and the emotions they evoke. Our core research question is: Can we create semantically rich embeddings to build a high-quality, scalable semantic search experience over a large music library?
**What it Does:** Our application allows users to search for a type of song or album using text, specify the number of songs or albums they’d like, and return the top k matches associated with their query.

We leverage 2 separate models, songs are retrieved by predicting the Spotify features associated with the song for a given text query and then returning the top k cosine similarities scores between the predicted Spotify embedding and the embeddings of all the songs in the database. When the user requests fewer than 20 songs, the song search model conducts a two-step evaluation. First, predicting a multiple of k  (k * rerank_factor) songs that best match the embedding of the query, and then making a batched API call to Google’s Gemini LLM to annotate the songs. It then re-embeds the text and once again compares the Gemini-embedding to the query to return the k-highest similarity songs. Our album search mechanism leverages an encoder model that was fine-tuned on album reviews and potential queries for that album to embed each album as a vector. Similar to the song model, the album model computes the cosine similarity scores between the user’s query embedding and the album embeddings (leveraging RAG) and returns the top k albums corresponding to the highest similarity scores. 

**Quick Start:**
- Clone GitHub Repository 
- To run album search model, 

- To run song search model, create .env file with GEMINI_API_KEY=’your_api_key’
- Run the command 'streamlit run app.py' in the terminal to launch song search app

**To Use Your Own Music Dataset**
- Export your Spotify data using Exportify 
For Song Search:
- Add csv file

For Album Search:



**Video Links:**


**Evaluation:**

**Individual Contributions:**
- Alana developed and trained the song search model and Maya developed and trained the album search model. They collaborated on data collection and techniques for cleaning and augmenting the dataset and brainstorming creative ways to both embed the songs/albums and predict the best matches. They also collaborated on the final project demo and write-up.
