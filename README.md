# Automated-Health-Responses
Beyond the catch-all classification of "chatbot", there are some different flavors: *sentence completion, Q/A, dialogue goal-oriented, VQA (or visual dialogue), negotiation, Machine Translation*


**<u>Description:</u> A prototype project for automated, physician-like responses to medical questions**

## Getting Started:
* Clone repository
*

### Data:
* QuickUMLS (https://github.com/Georgetown-IR-Lab/QuickUMLS). A package to make accessing a huge medical concept database, UMLS (https://www.nlm.nih.gov/research/umls/)
* Data is from a subReddit: AskDoc. Downloaded from Google's Big Query (https://bigquery.cloud.google.com/table/fh-bigquery:reddit_posts.full_corpus_201509?pli=1). Data was up to 04-2018.

### Research:
* **Seq2Seq**
  * TF Seq2Seq models: https://www.tensorflow.org/versions/r1.0/tutorials/seq2seq#tensorflow_seq2seq_library
  * Contextualizing Chatbots (creating intents): https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077
  * Dynamic Unrolling with Seq2Seq: https://github.com/ematvey/tensorflow-seq2seq-tutorials
  * Great *Alignment* explanation https://machinelearningmastery.com/how-does-attention-work-in-encoder-decoder-recurrent-neural-networks/
* **Papers**
  * Getting Started:
    * Continuing a conversation beyond simple Q/A *LEARNING THROUGH DIALOGUE INTERACTIONS BY ASKING QUESTIONS* https://arxiv.org/pdf/1612.04936.pdf
    * Kindof a long read but more detail than you'd get in the standard published paper *Teaching Machines to Converse* https://github.com/jiweil/Jiwei-Thesis/blob/master/thesis.pdf
    * **Retrieval-Based**
      * Main 2015 paper: https://arxiv.org/abs/1506.08909
        * Follow-up to 2015 paper using same dataset to benchmark some clustering and hierarchical methods: https://arxiv.org/pdf/1710.03430.pdf
      * Google's "Smart Reply" method for clustering email responses: http://www.kdd.org/kdd2016/papers/files/Paper_1069.pdf
      * Clustering by extracted entities: https://www.cs.utexas.edu/~ckcuong/Publications/Text%20Clustering%20with%20Named%20Entities.pdf
      * Chatbot with TF: https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077
      * Good reference list for reading: http://www.wildml.com/2016/04/deep-learning-for-chatbots-part-1-introduction/
      * **Chameleon paper:**http://www.cs.cornell.edu/~cristian/papers/chameleons.pdf
      * **Pandas pipelines:**https://medium.com/bigdatarepublic/integrating-pandas-and-scikit-learn-with-pipelines-f70eb6183696



### Notes About Approaches
* Dialogue systems (which include chatbots) generally can be classified under three categories:
  * The back-and-forth dialogue between algorithm and human
  * The frame-based, goal-oriented (think online help or call-routing)
  * The interactive Q/A system.
* The mechanism to generate the machine response to these systems can be generative (the machine comes up with its own response), or responsive (returns a pre-determined answer based on a classification). Most successful systems seem to have a combination of the two.
* Probably anyone with a smartphone searches online for something relating to their health. Although the first page, second page, wikipedia article or any one page may not be helpful, the process may itself reveal to the individual important insights: http://matjournals.in/index.php/JoWDWD/article/view/2334


### Notes about the dataset

*Warning:* Data from this forum may have more than its share of topics of a sexual nature. (Could easily be assumed because of the anonymity of reddit.)

Data is from when the subreddit was started (2014) to early 2018. There are approximately 30k threads, 109k responses.

### Data Journal
* **1st Iteration: Exploring model and modeling choices.**
  * Decided on architecture for data prep on first model for conversations. We frame the problem as bootstrapping responses to conversations in a general sense of someone who has a health-related question and someone who has some sort of knowledge on the subject. Given that there are multiple responses to potential the same question, the first pass is: someone asks a question on reddit thread and everyone post in that thread _not_ by the author is encoded as a response. This is a big consideration of what we could reasonably expect from a trained network. We are obviously over-sampling questions perhaps giving the network incentive to learn the most generic response to a random question
  * Found out reference code for the Tensorflow Seq2Seq model was depreciated because it uses *static unrolling:*
    * *Static unrolling involves construction of computation graph with a fixed sequence of time step. Such a graph can only handle sequences of specific lengths. One solution for handling sequences of varying lengths is to create multiple graphs with different time lengths and separate the dataset into this buckets.*
    * *Action:* Use **Dynamic Unrolling** Dynamic unrolling instead uses control flow ops to process sequence step by step. In TF this is supposed to more space efficient and just as fast. This is now a recommended way to implement RNNs.
* **2nd Iteration: Experimenting with generative model approach**
  * So far just using a word-level, teacher-forcing for 1 step ahead Seq2Seq is doing not well (currently based primarily off the reasonableness of responses to training set), but understandably at least. This is currently serving as a baseline when deciding further directions to pursue.
    * There are some issues with current data approach since the model is tending to generalize to a politically phrased response: "I'm not a doctor but"
      * **Q: Husband deteriorating before my eyes, doctors at a loss, no one will help; Reddit docs, I need you.**
      * **A: I don't think this is a single pain is not a doctor but I have a similar symptoms and the story**
  * As suspected, even with seq2seq at a word level, we are getting not so great results. Although have not trained on full dataset yet, there is a decided improvement when using less than 30 words for response. One option would be change pipeline and limit words and sentences. However I suspect the bigger issue is that many posts to initial post are not direct responses. Structuring data using as parent/post might be the right approach to try first.
* **3rd Iteration: Response Retrieval**
  * Altered dataset so each post that had a comment posted as reply is treated as direct response. So occasionally one comment may be both a query and a response. Test training at a word level without any cleansing of data lead to very poor results as expected.
  * Successfully implemented dual_encoder with large improvements over baseline:
    * *Training Notes:*
      * Substantial gains were made by:
        * Adding dropout of 0.5 for the hidden layer
        * Switching to GRU RNN vs LSTM
        * Halving the number of neurons for the 1 hidden layer. Incidentally, this also dramatically decreased training time due to decrease matrix computation.

    * Random Baseline:
      * Recall @ (1, 10): 0.100675
      * Recall @ (2, 10): 0.2
      * Recall @ (5, 10): 0.399273
    * TF-IDF baseline:
      * Recall @ (1, 10): 0.476141
      * Recall @ (2, 10): 0.570431
      * Recall @ (5, 10): 0.722859
    * Dual Encoder with 6B 300d Glove:
      * Recall @ (1,10): 0.61527
      * Recall @ (2,10) 0.77616
      * Recall @ (5,10) 0.944514
    * Dual Encoder with 840B 300d Glove:
      * Recall @ (1,10): 0.715415
      * Recall @ (2,10) 0.87425
      * Recall @ (5,10) 0.974819

    Of course since current implementation of the model is binary (predicting out of 10 possible choices whether the response is correct or not), it really only makes sense to pay attention to Recall@1.
* **4th Iteration: Improving Relevancy of what Response Retrieval is...well, retrieving**


### Future Work:
  * One big issue with trying to generated responses to queries is determining which are queries and which are responses. Using word embeddings and computing a similarity score using the Word Movers Algorithm, we can get very similar types of phrases to a type of query. Example below:
  * Examples of a seed question that could be classified as inquiring about further information:
    * Seed: *"Hey, how's your husband doing now? Hope everything is okay."*
    * *"So why are you posting on here then, if you had two 'real' doctors giving you advice? What answer are you looking for here? "*
    * *'How long ago did you change your diet, as in when did you have the kidney stones?'*
    * *'How old is your partner?\n\nDo you know her diagnosis (ie why they did her surgery)?',*

## Acknowledgements
