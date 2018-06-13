def load_sem_types(data_path='../data/SemGroups_2013.txt'):
    """
    Create a dictionary for to return the category and tag associated with a given semantic type tagged by QuickUMLS.
    """
    sem_type_dict = {}
    with open(data_path,'r') as f:
        for line in f:
            lines = line.replace('\n','').split('|')
            sem_type_dict[lines[2]] = {'Category':lines[1],'Tag':lines[3]}
    return sem_type_dict



def tag_utterances(id,txt,tagger,sems = load_sem_types()):
    """
    Using QuickUMLS (a wrapper for UMLS https://www.nlm.nih.gov/research/umls/), this function tags entity types and the start and end points in the utterance.
    
    id: The index number of the utterance.
    txt: Utterance
    tagger: QuickUMLS spacy entity tagger object.
    
    Input: A sequence (list) of utterances
    Output:
        
    """
    
    stpwds = set()
    # Run the document through the NLP pipeline
    doc = tagger.nlp(txt)
    
    # Create a list with the indeces of the stop words
    for token in doc:
        if token.is_stop:
            stpwds.add(token.idx)
    # Run the UMLS tagger        
    matches= tagger.match(txt, best_match=True, ignore_syntax=False)
    data = []
    
    # Reiterate through the different matches returned for an utterance
    for match in matches:
        semtypes = set()
        term = ''
        cui = ''
        ngram = ''
        similarity=0
        # For every match collect all the semantic types
        # keep only the term with the highest matching score (similarity)
    
        for m in match:
            for s in m['semtypes']:
                semtypes.add(s)
            if m['similarity'] > similarity:
                term = m['term']
                cui = m['cui']
                similarity=m['similarity']
                ngram = m['ngram']
                
        # Filter out terms shorter than 3 chars
        if len(term)<=2:
            continue
        # Filter out stop words
        if match[0]['start'] in stpwds:
            continue
            
        tmp=[]
        tmp.append(id)
        tmp.append(match[0]['start'])
        tmp.append(match[0]['end'])
        tmp.append(term.lower())
        tmp.append(cui)
        tmp.append(similarity)
        stypes_cat = set()
        

        for sem in semtypes:
            stypes_cat.add(sems[sem]['Tag'])

        tmp.append(stypes_cat)

        data.append(tmp)
    return data




import pandas as pd
import os
import sys
import glob

class DataPipeline:
    """
    Injest Reddit AskDocs Data
    """
    
    def __init__(self,comments_path='/',posts_path='/'):
        self.comments_path = comments_path
        self.posts_path = posts_path
        
        
    def load_comments(self):
        """
        Load comments from reddit dataset and process.
        """

        assert len(self.comments_path) > 1

        dataframe = pd.read_csv(self.comments_path,dtype={'body':str,'score_hidden':float},low_memory=False)

        # All posts with the same parent_id (following the "_") as the link_id are. 
        # E.g. If the link_id is "t3_827pgt", all parent_id's with "827pgt" are pointing towards that original post.
        dataframe['link_id_short'] = dataframe['link_id'].apply(lambda r: str(r).split('_')[1])
        dataframe['parent_id_short'] = dataframe['parent_id'].apply(lambda r: str(r).split('_')[1])
        # rename id column to be more transparent
        dataframe['post_id'] = dataframe['id']

        # Double-check that there is no uniqueness lost when eliminating the t-tags
        assert len(dataframe['link_id_short'].unique() == dataframe['link_id'].unique())
        assert len(dataframe['parent_id_short'].unique() == dataframe['parent_id'].unique())

        print('Comments Table Shape:',dataframe.shape)

        return dataframe
    
    def load_posts(self):
        """
        Load posts from reddit dataset and process.
        """
        dataframe = pd.read_csv(self.posts_path,low_memory=False)
        dataframe['link_id_short'] = dataframe['id']
        print('Posts table shape:',dataframe.shape)
        return dataframe
    
    def load_full_thread(self):
        """
        Loads a unified dataset of posts and comments tables. (comments tables do not contain original post)
        """
        
        # import data

        df_comments = self.load_comments()
        df_posts = self.load_posts()

        ## Get uniqueness
        # Uniqueness among posts
        comment_post_ids = [str(c).strip() for c in df_comments['link_id_short'].unique().tolist()]
        post_ids = [str(c).strip() for c in df_posts['id'].unique().tolist()]

        # Get set of ids that are in both tables:
        id_intersect = (set(post_ids) & set(comment_post_ids))
        print(len(id_intersect))

        # Create table with only those intersect ids
        df_intersect = df_posts.loc[df_posts['id'].isin(id_intersect)]
        df_intersect = df_intersect.rename(index=str, columns={"selftext": "body"}).copy()
        
        # Create indicator column for join to identify if corpus is original post
        df_intersect['is_thread_start'] = 1

        # Deterimine which columns would be helpful in final table
        columns_in_both = (set(df_comments.columns) & set(df_intersect.columns))
        columns_in_both.update(["title","url","over_18","is_thread_start"])
        columns_in_both = list(columns_in_both)

        # Get comments following threads that are in both posts and comments
        df_comments = df_comments.loc[df_comments['link_id_short'].isin(id_intersect)].copy()
        # Get final intersect table
        df_intersect = df_comments.append(df_intersect[columns_in_both]).copy()
        df_intersect = df_intersect.fillna(value={'is_thread_start':0}).copy()
        print('Final combined table shape:',df_intersect.shape)
        
        return df_intersect
    
    
class CalculateBleu(chainer.training.Extension):

    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self, model, test_data, key, batch=100, device=-1, max_length=100):
        self.model = model
        self.test_data = test_data
        self.key = key
        self.batch = batch
        self.device = device
        self.max_length = max_length

    def __call__(self, trainer):
        with chainer.no_backprop_mode():
            references = []
            hypotheses = []
            for i in range(0, len(self.test_data), self.batch):
                sources, targets = zip(*self.test_data[i:i + self.batch])
                references.extend([[t.tolist()] for t in targets])

                sources = [
                    chainer.dataset.to_device(self.device, x) for x in sources]
                ys = [y.tolist()
                      for y in self.model.translate(sources, self.max_length)]
                hypotheses.extend(ys)

        bleu = bleu_score.corpus_bleu(
            references, hypotheses,
            smoothing_function=bleu_score.SmoothingFunction().method1)
        chainer.report({self.key: bleu})