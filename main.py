import numpy as np
import pandas as pd
import scipy

from sklearn.feature_extraction.text import CountVectorizer

def read_ris(risFile):
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
    for i,k in enumerate(ris):
        if k[:2] == 'AB':
            abstract=''.join(ris[i:])
            ris[i] = abstract
            ris=ris[:(i+1)]
            return ris

    return ris
        
def parse_ris(ris):

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

    def __init__(self):
        self.P = None
        pass


    def fit(self,X):
        # compute P - probability matrix

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
                        


        self.vocabularyWords = list(pairs.keys())
        self.vocabularyWords.append('STOP')
        pairs['STOP'] = []
        self.vocabulary_ = {k:i for i,k in enumerate(self.vocabularyWords)}
        self.reverse_vocabulary_ = {i:k for i,k in enumerate(self.vocabularyWords)}
        N = len(self.vocabulary_)
        # now compute matrix
        self.P = scipy.sparse.csc_matrix((N,N))
        for k,i in self.vocabulary_.items():            
            ps = pd.Series(pairs[k]).groupby(pairs[k]).count()
            idx = np.in1d(self.vocabularyWords,ps.index)
            self.P[i,idx] = ps

    def simulate(self,startWord=None,maxWords=40):
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


                    
    def combinatorial_search(words):
        idx = np.in1d( list(self.vocabulary_.keys()), words)
        p = self.Ps[idx,:]+.0001
        logProb = np.log(p/p.sum())
        permIdx=permutation(idx)
        costs = [np.sum([logProb[permIdx[k],permIdx[k+1]]]) for permIdx in permutation(Idx) \
                     for k in permIdx]
                     
        minCostIndex = np.argmin(costs)
        
        minComb = permIdx[minCostIndex]

        return words[minComb]

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

    X = countVec.transform(abstracts)
    y = countVec.transform(titles)

    Z = X
    v = y

    mc = MarkovChain()
    
    mc.fit(titles)
    newTitle = ' '.join(mc.simulate('neuronal'))

