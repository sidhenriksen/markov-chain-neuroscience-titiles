import numpy as np
import pandas as pd
import scipy

from sklearn.feature_extraction.text import CountVectorizer

def read_ris(risFile):
    '''
    Reads an RIS file downloaded from sciencedirect.com.
    
    Parameters
    -----------
    risFile : str, path to RIS file

    Returns
    --------
    data : list, list of dictionaries, each corresponding to a paper
    
    '''

    with open(risFile,'r') as f:
       rawData = f.readlines()

    data = []
    for i,k in enumerate(rawData):
        if 'TY  ' in k and 'JOUR' in k:
            start = i

        if 'ER  -' in k:
            stop = i
            currentData = parse_ris(rawData[start:stop])
            data.append(currentData)

    return data

def join_abstract(ris):
    '''
    Joins abstract fields that are on separate lines
    '''
    for i,k in enumerate(ris):
        if k[:2] == 'AB':
            abstract=''.join(ris[i:])
            ris[i] = abstract
            ris=ris[:(i+1)]
            return ris

    return ris
        
def parse_ris(ris):
    '''
    Parses an appropriately partitioned RIS file
    
    Parameters
    -----------
    ris : str, segment of the overall RIS file corresponding to 

    Returns
    -------
    data : dict, parsed elements of the string into appropriate key-value pairs

    '''

    ris = join_abstract(ris)
    
    get_value = lambda x:x.split(' - ')[1].replace('\n','').replace('\r','')
    data = {'authors':[]}
    entryDict = {'T1':'title',
                     'AU':'authors',
                     'JO':'journal',
                     'AB':'abstract'}
    for k in ris:
        if len(k) < 2:
            continue

        if k[:2] in entryDict:
            currentField = entryDict[k[:2]]
            if k[:2] == 'AU':
                data[currentField].append(get_value(k))
            else:
                data[currentField] = get_value(k)

    return data


class MarkovChain:
    '''
    Implements a Markov chain using an sklearn-like interface
    
    '''

    def __init__(self):
        self.P = None
        pass


    def fit(self,X):
        '''
        Fits the Markov chain: generates a count matrix where
        the jth column of the ith row corresponds to the number of times
        word j was seen after having seen word i.

        Parameters
        -----------
        X : list, list of strings, each corresponding to a title/abstract, etc.
        
        Returns
        --------
        Nothing.

        '''

        # This first bit computes the pairwise probabilities
        pairs = {}
        for i in range(len(X)):
            x = X[i].replace(',','').replace(':',' :').lower().split(' ')
            for j,_ in enumerate(x):
                if j < (len(x)-1):
                    if x[j] not in pairs:
                        pairs[x[j]] = [x[j+1]]
                    else:
                        pairs[x[j]].append(x[j+1])
                        
                else:
                    if x[j] not in pairs:
                        pairs[x[j]] = ['STOP']
                        
                    else:
                        pairs[x[j]].append('STOP')
                        


        # Do some housekeeping
        # Append "STOP" which corresponds to stop seentence
        # Generate vocabularies, etc.        
        self.vocabularyWords = list(pairs.keys())
        self.vocabularyWords.append('STOP')
        pairs['STOP'] = []
        self.vocabulary_ = {k:i for i,k in enumerate(self.vocabularyWords)}
        self.reverse_vocabulary_ = {i:k for i,k in enumerate(self.vocabularyWords)}
        N = len(self.vocabulary_)
        
        # Now compute count matrix P 
        self.P = scipy.sparse.csc_matrix((N,N))
        for k,i in self.vocabulary_.items():            
            ps = pd.Series(pairs[k]).groupby(pairs[k]).count()
            idx = np.in1d(self.vocabularyWords,ps.index)
            self.P[i,idx] = ps

    def simulate(self,startWord=None,maxWords=25):
            '''
            Simulate a title with the Markov chain, using an opttonal
            start word.
            
            Parameters
            ----------
            startWord : str, optional. Default is None, i.e. random start word.
            maxWords : int, optional. Max number of words (default 40)

            Returns
            ------
            title : str, simulated MC title
            '''
            
            if startWord is None:
                    startWord = np.random.choice(self.vocabularyWords)
                    
            currentWord = startWord
            title = [startWord]
            for k in np.arange(maxWords):

                ps = self.P[self.vocabulary_[currentWord],:]

                idx = ps.nonzero()[1]

                N = np.array(ps[0,idx].todense())[0,:]
                candidateWords = {self.reverse_vocabulary_[j]:N[i] for i,j in enumerate(idx)}

                wordArray = []
                for word,n in candidateWords.items():
                    wordArray.extend([word]*np.int64(n))

                currentWord = np.random.choice(wordArray,1)[0]
                if currentWord=='STOP':
                    break
                
                title.append(currentWord)
                
            return title

if __name__ == "__main__":
    fileName = 'data/alldata.ris'

    data = read_ris(fileName)


    countVec = CountVectorizer()

    abstracts,titles = [],[]
    for k in data:
        if 'abstract' in k and 'title' in k:
            abstracts.append(k['abstract'])
            titles.append(k['title'])


    countVec.fit(abstracts)

    mc = MarkovChain()
    
    mc.fit(titles)

    for k in range(5):
        newTitle = ' '.join(mc.simulate(maxWords=30))
        print(newTitle)
        print('')

        

