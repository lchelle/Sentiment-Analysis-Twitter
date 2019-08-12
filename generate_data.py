#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from collections import defaultdict
from fnmatch import fnmatch

reactions = []
rumours_tweet_ids = []
n_rumours_tweet_ids = []

id_tweet_dict = dict()
id_reaction_dict = dict()

# tweet: [reaction1, reaction2]
tweetID_reactions_dict = defaultdict(list)


def main():
    root = "./sydney/"
    pattern = "*.json"
    print("Reading the twitter data")

    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name, pattern):
                if "/rumours/" in os.path.join(path, name):
                    if "source-tweets" in os.path.join(path, name):
                        with open(os.path.join(path, name)) as Jsfile:
                            data = json.load(Jsfile)
                            tweet_text = data["text"].encode("utf-8")
                            tweet_id = str(data["id"])

                            rumours_tweet_ids.append(tweet_id)
                            id_tweet_dict[tweet_id] = tweet_text

                    if "reactions" in os.path.join(path, name):
                        with open(os.path.join(path, name)) as Jrfile:
                            data = json.load(Jrfile)
                            reaction_text = data["text"].encode("utf-8")
                            reaction_id = str(data["id"])
                            source_tweet_id = str(data["in_reply_to_status_id"])
                            id_reaction_dict[reaction_id] = reaction_text
                            tweetID_reactions_dict[source_tweet_id].append(reaction_text)

                if "non-rumours" in os.path.join(path, name):
                    if "source-tweets" in os.path.join(path, name):
                        with open(os.path.join(path, name)) as Jsfile:
                            data = json.load(Jsfile)
                            tweet_text = data["text"].encode("utf-8")
                            ntweet_id = str(data["id"])

                            n_rumours_tweet_ids.append(ntweet_id)
                            id_tweet_dict[ntweet_id] = tweet_text

                    if "reactions" in os.path.join(path, name):
                        with open(os.path.join(path, name)) as Jrfile:
                            data = json.load(Jrfile)
                            nreaction_text = data["text"].encode("utf-8")
                            nreaction_id = str(data["id"])
                            nsource_tweet_id = str(data["in_reply_to_status_id"])
                            id_reaction_dict[nreaction_id] = nreaction_text
                            tweetID_reactions_dict[nsource_tweet_id].append(nreaction_text)
    print("Writing tweets to the file")
    try:
        os.remove("tweets_sydney.csv")
    except OSError:
        pass
    count = 0
    with open("tweets_sydney.csv", 'w') as i_file:
        i_file.write('TWEET_ID,TWEET,RUMOUR_NONRUMOR,RETWEETS\n')
        for id, tweet in id_tweet_dict.items():
            for reaction in tweetID_reactions_dict[id]:
                if id:
                    i_file.write(id + "," +
                                 tweet.replace('\n', '')
                                 .replace('\r', '')
                                 .replace(',', '')
                                 .replace('"', '')
                                 .replace('“', '')
                                 .replace('”', '') + ",")
                if id in rumours_tweet_ids:
                    i_file.write("0" + ",")
                if id in n_rumours_tweet_ids:
                    i_file.write("1" + ",")
                i_file.write(reaction.replace('\n', '')
                             .replace('\r', '')
                             .replace(',', '')
                             .replace('"', '')
                             .replace('“', '')
                             .replace('”', '') + "\n")
                count += 1

    print("Rumours source tweets: " + str(len(rumours_tweet_ids)))
    print("Non-Rumours source tweets: " + str(len(n_rumours_tweet_ids)))
    print("Total retweets: " + str(len(id_reaction_dict)))
    print("Total datalines: " + str(count))


if __name__ == "__main__":
    main()
