# SpotiSearch

### Goal
This project aims to build a semantic search engine for albums and songs that retrieves albums and tracks based on their vibe and the emotions they evoke. Our core research question is: Can we create semantically rich embeddings to build a high-quality, scalable semantic search experience over a large music library?

### What it Does
Our application allows users to search for a type of song or album using text, specify the number of songs or albums they’d like, and return the top k matches associated with their query.
We leverage 2 separate models: songs are retrieved by predicting the Spotify features associated with the song for a given text query and then returning the top k cosine similarities scores between the predicted Spotify embedding and the embeddings of all the songs in the database. When the user requests fewer than 20 songs, the song search model conducts a two-step evaluation. First, predicting a multiple of k  (k * rerank_factor) songs that best match the embedding of the query, and then making a batched API call to Google’s Gemini LLM to annotate the songs. It then re-embeds the text and once again compares the Gemini-embedding to the query to return the k-highest similarity songs. Our album search mechanism leverages an encoder model that was fine-tuned on album reviews and potential queries for that album to embed each album as a vector. Similar to the song model, the album model computes the cosine similarity scores between the user’s query embedding and the album embeddings (leveraging RAG) and returns the top k albums corresponding to the highest similarity scores. 

### Quick Start
- Clone GitHub Repository 
- Install requirements from txt file
- To run album search model,
- To run song search model, create .env file with GEMINI_API_KEY=’your_api_key’
- Run the command 'streamlit run app.py' in the terminal to launch song search app

#### To Use Your Own Music Dataset
- Export your Spotify data using Exportify 
##### For Song Search:
- Add csv file to data folder
- Run the main method in extract_embeddings_from_csv.py to compute and store the song embeddings
- Update the file path(s) in config.json to your data file path and newly computed embeddings file path
- Relaunch app.py!

##### For Album Search:


### Video Links


### Evaluation
#### Song Search
1. During training, we were able to achieve an error rate of 59% on our test set
2. Although it is challenging to evaluate the recall or precision of the generative search model, the songs returned for a variety of queries tend to align with the prompt. Although the search mechanism does not perform perfectly when the prompt is out of distribution, many prompts tend to produce plausible search results.

**For example:**
1. The query: "Synth disco electric" returns these top 3 songs (k = 30):
   2. Shutdown Sequence - Willix (score=0.9822)
   3. Sina - Lakwister, Aweko Brian, Moses rallo (score=0.9752)
   4. Udokotela - Muzari (score=0.9723)

2. The query: "chill, slow, studying songs" returns these top 3 songs (k = 30):
   3. 10,000 Hours (with Justin Bieber) - Dan + Shay, Justin Bieber (score = 0.9151)
   4. Nothing Breaks Like a Heart (feat. Miley Cyrus) - Mark Ronson, Miley Cyrus (score = 0.8954)
   5. Somebody That I Used To Know - Gotye, Kimbra (score=0.8848)

Upon listening, the songs align well with the query and are stronger matches than had we randomly selected 3 songs from the database for both queries

#### Album Search

### Individual Contributions
Alana developed and trained the song search model and Maya developed and trained the album search model. They collaborated on data collection and techniques for cleaning and augmenting the dataset and brainstorming creative ways to both embed the songs/albums and predict the best matches. They also built the web app together using Streamlit. They also collaborated on the final project demo and write-up. 
