# SpotiSearch

### Goal
This project aims to build a semantic search engine for albums and songs that retrieves albums and tracks based on their vibe and the emotions they evoke. Our core research question is: Can we create semantically rich embeddings to build a high-quality, scalable semantic search experience over a large music library?

### What it Does
Our application allows users to search for a type of song or album using text, specify the number of songs or albums they’d like, and return the top k matches associated with their query.
We leverage 2 separate models: songs are retrieved by predicting the Spotify features associated with the song for a given text query and then returning the top k cosine similarities scores between the predicted Spotify embedding and the embeddings of all the songs in the database. When the user requests fewer than 20 songs, the song search model conducts a two-step evaluation. First, predicting a multiple of k  (k * rerank_factor) songs that best match the embedding of the query, and then making a batched API call to Google’s Gemini LLM to annotate the songs. It then re-embeds the text and once again compares the Gemini-embedding to the query to return the k-highest similarity songs. Our album search mechanism leverages an encoder model that was fine-tuned on album reviews and matching queries for that album to embed each album as a vector. Similar to the song model, the album model computes the cosine similarity scores between the user’s query embedding and the album embeddings (leveraging RAG) and returns the top k albums corresponding to the highest similarity scores. 

### Quick Start
- Clone GitHub Repository 
- Install requirements from txt file
- Navigate to our webapp linked [here](https://spotisearch.streamlit.app/)
- Or, run the command 'streamlit run app.py' from the root of the repository in the terminal to launch song search app


### Video Links
1. [Demo Video](https://www.youtube.com/watch?v=CCkN5tFVGAg)

2. [Technical Walkthrough](https://www.youtube.com/watch?v=1IsiDmNnPcs)


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
1. During training, we fine tuned a stage 1 contrastive model with no hard negatives measured by MultipleNegativesRankingLoss which uses many negatives per batch and a log-softmax over similarities. This basic model achieved a trianing loss of .678, which is high because the model uses all other positives in the batch as negatives and is only used to get a basic structure of global relations in our custom embedding space.
2. We used a stage 2 triplet contrastive model, which utilizes hard negative triplets to further train the stage 1 model and TripletLoss to measure loss. TripletLoss compares only one positive and one negative per triplet and achieved a training loss of 0.137, which shows strong fine-grained discrimination in our model as the positives and negatives are well-separated.
3. For the retrieval itself using the stage 2 model, we used recall to measure the performance of our embedding model. We achieved Recall@1: 0.4404, Recall@5: 0.5923, and Recall@10: 0.6519. 44% Recall@1 means nearly half the queries return the exact correct album as the top result, 59% @5 means over half the queries return the exact correct album in the top 5 resuts, and 65% @10 means that the exact album was returned in the top 10 results 65% of the time. For a mood browsing type interface, this is excellent because users rarely expect one perfect answer.

**For example:**
1. The query: dreamy late-night
   2. You Are Here — +/- 
   3. Carrion EP — Jana Hunter | Folk/Country
   4. One in an Infinity of Ways — Ammoncontact | Electronic / Jazz / Rap
   5. Catch That Totem! (1998-2005) — Alog | Electronic / Jazz / Rock
   6. A Little Big — Bobby and Blumm 

#### To Use Your Own Music Dataset
1. Export your Spotify data using [Exportify](https://exportify.net/) 

##### For Song Search:
2. Add csv file to data folder
3. Run the main method in extract_embeddings_from_csv.py to compute and store the song embeddings
4. Update the file path(s) in config.json to your data file path and newly computed embeddings file path
5. Relaunch app.py!

##### For Album Search:
2. Navigate to create_my_albums.py and change FILE_MYALBUMS to your data file and run the file
3. In music_loader.py, login to Huggingface and retrieve an API token
4. Update the dataframe on line 14 to point to your albums datafile
5. Update the embeddings file path in app.py on line 18 to point to your newly saved embeddings
5. Relaunch the app!

### Individual Contributions
Alana developed and trained the song search model and Maya developed and trained the album search model. They collaborated on data collection and techniques for cleaning and augmenting the dataset and brainstorming creative ways to both embed the songs/albums and predict the best matches. They also built the web app together using Streamlit. They also collaborated on the final project demo and write-up. 
