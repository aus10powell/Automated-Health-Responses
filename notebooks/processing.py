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


def load_comments(path_to_comments=''):
    """
    Load comments from reddit dataset and process
    """
    import pandas as pd
    
    assert len(path_to_comments) > 1
    
    dataframe = pd.read_csv(path_to_comments,dtype={'body':str,'score_hidden':float},low_memory=False)
    
    # All posts with the same parent_id (following the "_") as the link_id are. 
    # E.g. If the link_id is "t3_827pgt", all parent_id's with "827pgt" are pointing towards that original post.

    dataframe['link_id_short'] = dataframe['link_id'].apply(lambda r: str(r).split('_')[1])
    dataframe['parent_id_short'] = dataframe['parent_id'].apply(lambda r: str(r).split('_')[1])

    # Double-check that there is no uniqueness lost when eliminating the t-tags
    assert len(dataframe['link_id_short'].unique() == dataframe['link_id'].unique())
    assert len(dataframe['parent_id_short'].unique() == dataframe['parent_id'].unique())
    
    print('Shape:',dataframe.shape)
    
    return dataframe
    