# Automated-Health-Responses
Beyond the catch-all classification of "chatbot", there are some different flavors: *sentence completion, Q/A, dialogue goal-oriented, VQA (or visual dialogue), negotiation, Machine Translation*


**<u>Description:</u> A prototype project for automated, physician-like responses to medical questions**



### Data/Resources:
* QuickUMLS (https://github.com/Georgetown-IR-Lab/QuickUMLS). A package to make accessing a huge medical concept database, UMLS (https://www.nlm.nih.gov/research/umls/)
* Data is from a subReddit: AskDoc. Downloaded from Google's Big Query (https://bigquery.cloud.google.com/table/fh-bigquery:reddit_posts.full_corpus_201509?pli=1). Data was up to 04-2018.
* Seq2Seq
  * TF Seq2Seq models: https://www.tensorflow.org/versions/r1.0/tutorials/seq2seq#tensorflow_seq2seq_library
  * Contextualizing Chatbots (creating intents): https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077
  * Dynamic Unrolling with Seq2Seq: https://github.com/ematvey/tensorflow-seq2seq-tutorials
  * Great *Alignment* explanation https://machinelearningmastery.com/how-does-attention-work-in-encoder-decoder-recurrent-neural-networks/



### Notes about the dataset

*Warning:* Data from this forum may have more than its share of topics of a sexual nature. (Could easily be assumed because of the anonymity of reddit.)

Data is from when the subreddit was started (2014) to early 2018. There are approximately 30k threads, 109k responses.

### Data Journal
* Week of June 1:
  * Decided on architecture for data prep on first model for conversations. We frame the problem as bootstrapping responses to conversations in a general sense of someone who has a health-related question and someone who has some sort of knowledge on the subject. Given that there are multiple responses to potential the same question, the first pass is: someone asks a question on reddit thread and everyone post in that thread _not_ by the author is encoded as a response. This is a big consideration of what we could reasonably expect from a trained network. We are obviously over-sampling questions perhaps giving the network incentive to learn the most generic response to a random question
  * Found out reference code for the Tensorflow Seq2Seq model was depreciated because it uses *static unrolling:*
    * *Static unrolling involves construction of computation graph with a fixed sequence of time step. Such a graph can only handle sequences of specific lengths. One solution for handling sequences of varying lengths is to create multiple graphs with different time lengths and separate the dataset into this buckets.*
    * *Action:* Use **Dynamic Unrolling** Dynamic unrolling instead uses control flow ops to process sequence step by step. In TF this is supposed to more space efficient and just as fast. This is now a recommended way to implement RNNs.
* Week of June 10:
  * So far just using a character-level, teacher-forcing for 1 step ahead Seq2Seq is doing reasonably well (currently based primarily off the reasonableness of responses to training set). This is currently serving as a baseline when deciding further directions to pursue.
    * There are some issues with current data approach since the model is tending to generalize to a politically phrased response: "I'm not a doctor but"
      * **Q: Husband deteriorating before my eyes, doctors at a loss, no one will help; Reddit docs, I need you.**
      * **A: I don't think this is a single pain is not a doctor but I have a similar symptoms and the story**
      * **Q: pleomorphic adenoma and a little scared**
      * **A: I don't think this is a single pain is not a doctor but I have a similar symptoms and the story**
      * **Q: I think I have Strep Throat. I do not have insurance and I cannot afford to go to the doctor.**
      * **A: I don't think this is a single pain is not a doctor but I have a similar symptoms and the story**
  * As suspected, even with seq2seq at a word level, we are getting not so great results. Although have not trained on full dataset yet, there is a decided improvement when using less than 30 words for response. One option would be change pipeline and limit words and sentences. However I suspect the bigger issue is that many posts to initial post are not direct responses. Structuring data using as parent/post might be the right approach to try first.
