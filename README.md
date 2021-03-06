# Spotify_Hybrid-Recommender-System
Data collected from Spotify API, Using both Content-Based &amp; Machine Learning Recommendation



# Modelling
**Content-Based Filtering**

- Cosine similarity + Euclidean Distance
- In order to solve "Cold Start" problem, few questions are given to the users. This allows me to get simple information about users, which helps recommending songs at the start of the service. 



**K-Means Clustering Recommendation**
- Since our data does not have any labels such as whether user prefers song or not, my idea was to cluster the track into different groups by using unsupervised learning. 
- Thus, K-Means Clustering was involved and our track is classified into 6 different groups.
- If I get the user's playlist, this system classifies which cluster does user's playlist belongs to and recommend similar songs from the same cluster.




# Features Info

**user_id:** spotify users unique id

**Track Name:** Name of the track

**Artist Name(s)**: Artist name of the track

**Album Name:** Track belongs to this album name

**Album Release Date:** Releasing date of album

**Track Number:** Unique track number of each song track

**Track Duration (ms):** Duration of each song track

**Explicit:** The explicit logo is applied when the lyrics or content of a song or a music video contain one or more of the following criteria which could be considered offensive or unsuitable for children

**Popularity:** For how many times song has been played

**Added By:** who added the track in playlist

**Added At:** Time at which a song added to the playlist

**Danceability:** it measure describes how suitable a track is for dancing

**Energy:** represents a perceptual measure of intensity and activity.

**Key:** the track is in. Integers map to pitches using standard Pitch Class notation. Valence measures from 0.0 to 1.0 describing the musical positiveness conveyed by a track

**Loudness:** Loudness of a track in decibels(dB).

**Mode:** Mode indicates the modality(major or minor) of the song

**Speechiness:** Speechiness detects the presence of spoken words in a track

**Acousticness**: Acosticness confidence measure from 0.0 to 1.0 of whether the track is acoustic

**Instrumentalness:** Instrumentalness predicts whether a track contains vocals or not

**Liveness**: it detects the presence of an audience in the recording

**Valence:** A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track

**Tempo:** Tempo is in beats per minute (BPM)

**Time Signature:** it is an estimated overall time signature of a track
