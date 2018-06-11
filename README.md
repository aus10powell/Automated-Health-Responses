# Automated-Health-Responses

**<u>Description:</u> A prototype project for automated, physician-like responses to medical questions**



### Data/Resources:
* QuickUMLS (https://github.com/Georgetown-IR-Lab/QuickUMLS). A package to make accessing a huge medical concept database, UMLS (https://www.nlm.nih.gov/research/umls/)
* Data is from a subReddit: AskDoc. Downloaded from Google's Big Query (https://bigquery.cloud.google.com/table/fh-bigquery:reddit_posts.full_corpus_201509?pli=1). Data was up to 04-2018.
* Tensorflow (https://www.tensorflow.org/)

### Notes about the dataset
* Threads may have more than their share of topics of a sexual nature. Could easily be assumed because of the anonymity of reddit.

### Data Journal
* Week of June 1:
  * Decided on architecture for data prep on first model for conversations. We frame the problem as bootstrapping responses to conversations in a general sense of someone who has a health-related question and someone who has some sort of knowledge on the subject. Given that there are multiple responses to potential the same question, the first pass is: someone asks a question on reddit thread and everyone post in that thread _not_ by the author is encoded as a response. This is a big consideration of what we could reasonably expect from a trained network. We are obviously over-sampling questions perhaps giving the network incentive to learn the most generic response to a random question
  * Found out reference code for the Tensorflow Seq2Seq model was depreciated because it uses *static unrolling:*
    * *Static unrolling involves construction of computation graph with a fixed sequence of time step. Such a graph can only handle sequences of specific lengths. One solution for handling sequences of varying lengths is to create multiple graphs with different time lengths and separate the dataset into this buckets.*
    * *Action:* Use **Dynamic Unrolling** Dynamic unrolling instead uses control flow ops to process sequence step by step. In TF this is supposed to more space efficient and just as fast. This is now a recommended way to implement RNNs.
* Week of June 10:
  * So far just using a character-level, teacher-forcing for 1 step ahead Seq2Seq is doing reasonably well (currently based primarily off the reasonableness of responses to training set). This is currently serving as a baseline when deciding further directions to pursue.  
