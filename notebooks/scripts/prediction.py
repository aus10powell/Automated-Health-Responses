from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
# Create pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from collections import defaultdict
import time
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd


def load_data():
    path_to_data = '../../data/reddit_comments_askDocs_2014_to_2018_03.gz'
    df = pd.read_csv(path_to_data,dtype={'body':str,'score_hidden':float})
    print('Shape',df.shape)
    df.head(2)

    df.dropna(subset=['body'],inplace=True)
    df['body'] = df['body'].astype(str)

    # Optional remove all strings where no/little response
    # df = df.loc[df['body'].apply(lambda r: len(str(r))> 2)]

    df['is_clinician'] = df['author_flair_text'].apply(lambda r: 0 if r =='This user has not yet been verified.' else 1)

    df['tokenized_sents'] = df['body'].apply(lambda row: str(row).strip().replace('\n','').lower().split(' '))

    return df

if __name__ == '__main__':
    df = load_data()


    text_clf = Pipeline([('vect', CountVectorizer(lowercase=True,ngram_range=(1,2),min_df=10)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB())
                        ])

    X_train, X_test, y_train, y_test = train_test_split(df['body'],df['is_clinician'] , test_size=0.2, random_state=329)

    print('training model...')
    print(X_train.shape)
    scores = cross_val_score(text_clf, X_train, y_train, cv=6,n_jobs=6,scoring='f1_macro')
    print("F1 macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
