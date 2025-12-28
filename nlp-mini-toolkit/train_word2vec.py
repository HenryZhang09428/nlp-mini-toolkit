from gensim.models import Word2Vec
import os
import nltk
import string
import numpy as np

tokeniser = nltk.tokenize.TreebankWordTokenizer()
stopwords = frozenset(nltk.corpus.stopwords.words("english"))
trans_table = str.maketrans(dict.fromkeys(string.punctuation))


def process_training_data(file_name, remove_stopwords=False):
    """
    Processes a text file that contains a collection of sentences.
    Performs tokenisation and optionally stop word removal.
    Returns a list of sentences, where each sentence is a list of tokens.
    Args:
        file_name (str): path to a text file, where each line of the file is a sentence
        remove_stopwords (bool): True if stop words are to be removed and False otherwise
    Returns:
        list(list(str)): list of sentences, where each sentence is a list of tokens
    """
    sentences = []
    
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            
            raw = raw.lower()
            raw = raw.translate(trans_table)
            tokens = tokeniser.tokenize(raw)
            tokens = [t for t in tokens if t.strip()]
            
            if remove_stopwords:
                tokens = [t for t in tokens if t not in stopwords]
            
            if len(tokens) > 0:
                sentences.append(tokens)
    
    return sentences



def train_model(sentences, window, seed):
    """
    Trains a word2vec model using the given sentences and the hyperparameters given.
    Args:
        sentences (list(list(str))): training sentences
        window: the size of the context windows used to train the model
        seed: seed of the random number generator used to initialise the word embeddings
    Returns:
        the trained word2vec model (which is a Word2Vec object)
    """
    w2v_model = Word2Vec(
        sentences,
        vector_size=200,
        window=window,
        min_count=10,
        sg=1,
        hs=0,
        negative=10,
        epochs=5,
        workers=1,
        seed=seed
    )
    return w2v_model



if __name__ == '__main__':
    file_name = os.path.join("data", "w2v_training_data.txt")
    
    configs = [
        {'window': 2, 'remove_stopwords': False, 'seed': 1, 'save_name': 'word2vec_M1.model'},
        {'window': 2, 'remove_stopwords': False, 'seed': 2, 'save_name': 'word2vec_M2.model'},
        {'window': 5, 'remove_stopwords': False, 'seed': 1, 'save_name': 'word2vec_M3.model'},
        {'window': 5, 'remove_stopwords': True, 'seed': 1, 'save_name': 'word2vec_M4.model'}
    ]
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Training Model: {config['save_name']}")
        print(f"Configuration: window={config['window']}, remove_stopwords={config['remove_stopwords']}, seed={config['seed']}")
        
        sentences = process_training_data(file_name, config['remove_stopwords'])
        print(f"Number of sentences: {len(sentences):,d}")
        
        if len(sentences) > 0:
            avg_length = sum(len(s) for s in sentences) / len(sentences)
            print(f"Average sentence length: {avg_length:.2f} tokens")
        
        w2v_model = train_model(sentences, config['window'], config['seed'])
        print(f"Vocabulary size: {len(w2v_model.wv):,d}")
        
        w2v_model.save(config['save_name'])
        print(f"Model saved to: {config['save_name']}")
        
        if 'baseball' in w2v_model.wv and 'basketball' in w2v_model.wv and 'computer' in w2v_model.wv:
            baseball_vec = w2v_model.wv['baseball']
            basketball_vec = w2v_model.wv['basketball']
            computer_vec = w2v_model.wv['computer']
            
            sim_baseball_basketball = w2v_model.wv.similarity('baseball', 'basketball')
            sim_baseball_computer = w2v_model.wv.similarity('baseball', 'computer')
            
            print(f"Cosine similarity (baseball, basketball): {sim_baseball_basketball:.4f}")
            print(f"Cosine similarity (baseball, computer): {sim_baseball_computer:.4f}")
        
        if 'would' in w2v_model.wv:
            sims = w2v_model.wv.most_similar('would', topn=20)
            print(f"Top-20 words most similar to 'would': {sims}")
        
        if 'greece' in w2v_model.wv:
            sims = w2v_model.wv.most_similar('greece', topn=20)
            print(f"Top-20 words most similar to 'greece': {sims}")
    
    print(f"\n{'='*60}")
    print("Cross-Model Cosine Similarity (Task C)")
    print(f"{'='*60}")
    
    m1 = Word2Vec.load('word2vec_M1.model')
    m2 = Word2Vec.load('word2vec_M2.model')
    
    if 'baseball' in m1.wv and 'basketball' in m2.wv:
        v1 = m1.wv['baseball']
        v2 = m2.wv['basketball']
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        cross_cos = float(np.dot(v1, v2))
        print(f"[C] Cross-model cosine (M1:baseball, M2:basketball): {cross_cos:.4f}")
        print(f"Note: This value is typically low and not meaningful because M1 and M2")
        print(f"      have different vector spaces due to different random seeds.")
        print(f"      The embeddings are not aligned across models.")
    else:
        print("[C] One of the tokens missing for cross-model cosine.")