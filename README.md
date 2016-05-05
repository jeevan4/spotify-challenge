**Spotify Data Challenge**

Dataset link:

https://storage.googleapis.com/ml_take_home/data_sample.tgz

Dataset description:

*end_song_sample.csv*

1. ms_played -- the amount of time the user listened to this track, in milliseconds
2. context -- the UI context the track was played from (e.g. playlist or artist page)
3. track_id -- the random UUID for the track
4. product -- the product status (e.g. free or paid)
5. end_timestamp -- the Epoch timestamp that marks the end of the listen
6. user_id -- the anonymous, random UUID of the user

*user_data_sample.csv*

1. gender -- the gender of the user (male or female)
2. age_range -- a bucketed age of the user
3. country -- the country where the user registered
4. acct_age_weeks -- the age of the user's account in weeks as of Oct 14th, 2015
5. user_id -- the anonymous, random UUID of the user
6. 

**Analysis:**

Found Male vs Female listening patterns

Breaked user listening into sessions

Looked for correlations between user demographic features (or their behavior) and their overall listening, or their average session lengths

Found a group of user categories that delineates some interesting or useful behavior traits. Here I showed user behaviour once they become Premium subscribers

