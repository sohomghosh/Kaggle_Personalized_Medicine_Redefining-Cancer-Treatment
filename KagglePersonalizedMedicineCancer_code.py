import pandas as pd
import numpy as np
import collections
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.preprocessing import LabelEncoder
from gensim.models import word2vec
import fasttext
import logging
import os
import xgboost as xgb
import re
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer

training_text=pd.read_csv("training_text",sep='\|\|',header=None,names=["ID","Text"])
training_text=training_text.drop(training_text.index[[0]])
training_variants=pd.read_csv("training_variants",header=None,names=["ID","Gene","Variation","Class"])
training_variants=training_variants.drop(training_variants.index[[0]])
train=pd.merge(training_variants,training_text,on='ID',how='inner')


test_text=pd.read_csv("test_text",sep='\|\|',header=None,names=["ID","Text"])
test_text=test_text.drop(test_text.index[[0]])
test_variants=pd.read_csv("test_variants",header=None,names=["ID","Gene","Variation","Class"])
test_variants=test_variants.drop(test_variants.index[[0]])
test=pd.merge(test_text,test_variants,on='ID',how='inner')


train_test=train.append(test)

######################################################################################################################################
##############################################Exploratory Data Analysis###############################################################
######################################################################################################################################

classwise_text=train.groupby('Class',as_index=False).apply(lambda x: ','.join(x.Text)).reset_index()

for i in range(0,classwise_text.shape[0]):
	wrds_to_replace=['et al','wild type','cell line','wild','type','amino acid','amino','acid','mutation']
	text=str(classwise_text.iloc[i,1])
	for j in wrds_to_replace:
		text=text.replace(j,'')
	wordcloud = WordCloud().generate(text)
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	#plt.show()
	plt.savefig('wordcloud'+str(i)+'.png',dpi=1000)
	plt.clf()

######################################################################################################################################
##############################################Feature Create: W2V#####################################################################
######################################################################################################################################


sentences_split=[re.split('\W', i) for i in train_test['Text']]
model_w2v = word2vec.Word2Vec(sentences_split, size=40,min_count =1, window=3, workers =-1,sample=1e-5)


features_sent = np.zeros(shape=(0,40))
for i in sentences_split:
	su=np.zeros(shape=(40))
	for j in i:
		k=np.array(model_w2v.wv[j])
		su=su+k
		#print(su)
	features_sent=np.vstack([features_sent, su])


np.savetxt("features_sent.csv", features_sent, delimiter=",")

######################################################################################################################################
##############################################Feature Create: FastText################################################################
######################################################################################################################################

train_test['Text'].to_csv('train_test_text.csv',index=False)

#For SkipGram
model_sk = fasttext.skipgram('train_test_text.csv', 'model')
print(model_sk['king'])
#get vectors for king
'''
input_file     training file path (required)
output         output file path (required)
lr             learning rate [0.05]
lr_update_rate change the rate of updates for the learning rate [100]
dim            size of word vectors [100]
ws             size of the context window [5]
epoch          number of epochs [5]
min_count      minimal number of word occurences [5]
neg            number of negatives sampled [5]
word_ngrams    max length of word ngram [1]
loss           loss function {ns, hs, softmax} [ns]
bucket         number of buckets [2000000]
minn           min length of char ngram [3]
maxn           max length of char ngram [6]
thread         number of threads [12]
t              sampling threshold [0.0001]
silent         disable the log output from the C++ extension [1]
encoding       specify input_file encoding [utf-8]
'''


#For CBOW
model_cbow = fasttext.cbow('train_test_text.csv', 'model',dim=40)
print(model_cbow['king'])
#get vectors for king
'''
input_file     training file path (required)
output         output file path (required)
lr             learning rate [0.05]
lr_update_rate change the rate of updates for the learning rate [100]
dim            size of word vectors [100]
ws             size of the context window [5]
epoch          number of epochs [5]
min_count      minimal number of word occurences [5]
neg            number of negatives sampled [5]
word_ngrams    max length of word ngram [1]
loss           loss function {ns, hs, softmax} [ns]
bucket         number of buckets [2000000]
minn           min length of char ngram [3]
maxn           max length of char ngram [6]
thread         number of threads [12]
t              sampling threshold [0.0001]
silent         disable the log output from the C++ extension [1]
encoding       specify input_file encoding [utf-8]
'''


######Need to check number of dimensions
features_sent_ft = np.zeros(shape=(0,40))
for i in sentences_split:
	su=np.zeros(shape=(40))
	for j in i:
		k=np.array(model_sk[j])
		su=su+k
		#print(su)
	features_sent_ft=np.vstack([features_sent_ft, su])


np.savetxt("features_sent_fast_text.csv", features_sent_ft, delimiter=",")

#gensim.models.wrappers.fasttext.FastText(sentences=None, size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=<built-in function hash>, iter=5, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000, compute_loss=False)
#Link: https://radimrehurek.com/gensim/models/wrappers/fasttext.html

######################################################################################################################################
##############################################Feature Create: TFIDF###################################################################
######################################################################################################################################

sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=.01, use_idf=True, smooth_idf=False, sublinear_tf=True)

sklearn_representation = sklearn_tfidf.fit(train_test['Text'])
train=pd.DataFrame(sklearn_tfidf.transform(train['Text']).todense())
test=pd.DataFrame(sklearn_tfidf.transform(test['Text']).todense())



######################################################################################################################################
############################################## Modeling ###################################################################
######################################################################################################################################



train_text_features=features_sent[0:3321,:]
test_text_features=features_sent[3321:,:]

train_text_features_df=pd.DataFrame(train_text_features)
test_text_features_df=pd.DataFrame(test_text_features)

train_text_features_df['Variation']=train['Variation']
test_text_features_df['Variation']=test['Variation']

train_text_features_df['Gene']=train['Gene']
test_text_features_df['Gene']=test['Gene']

train_text_features_df['Class']=train['Class']


for f in ['Variation','Gene']:#Add all categorical features in the list
    lbl = LabelEncoder()
    lbl.fit(list(train_text_features_df[f].values))
    train_text_features_df[f] = lbl.transform(list(train_text_features_df[f].values))

for f in ['Variation','Gene']:#Add all categorical features in the list
    lbl = LabelEncoder()
    lbl.fit(list(test_text_features_df[f].values))
    test_text_features_df[f] = lbl.transform(list(test_text_features_df[f].values))


X_train=train_text_features_df.sample(frac=0.80, replace=False)
X_train['Class']=pd.to_numeric(pd.Series(X_train['Class']),errors='coerce')

X_valid=pd.concat([train_text_features_df, X_train]).drop_duplicates(keep=False)
X_valid['Class']=pd.to_numeric(pd.Series(X_valid['Class']),errors='coerce')

X_test=test_text_features_df


features=list(set(train_text_features_df.columns)- set(['Class']))

X_train['Class']=X_train['Class']-1
dtrain = xgb.DMatrix(X_train[features], X_train['Class'], missing=np.nan)
dvalid = xgb.DMatrix(X_valid[features],missing=np.nan)
dtest = xgb.DMatrix(X_test[features], missing=np.nan)

nrounds = 260
watchlist = [(dtrain, 'train')]
params = {"objective": "multi:softmax","booster": "gbtree", "nthread": 4,"num_class": 9, "silent": 1,"eta": 0.08, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7,"min_child_weight": 1,"seed": 2016, "tree_method": "exact"}
bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=20)


valid_preds = bst.predict(dvalid)+1

###valid_prob = 

confusion_matrix(valid_preds,X_valid['Class'])

# array([[ 69,   3,   1,  13,   4,   4,   4,   0,   0],
#        [  1,  45,   0,   2,   2,   3,  19,   0,   0],
#        [  0,   0,   8,   0,   0,   0,   3,   0,   0],
#        [ 22,   2,   3,  98,   8,   5,   6,   0,   2],
#        [  7,   2,   1,   3,  18,   3,   1,   0,   0],
#        [  4,   1,   0,   0,   4,  34,   2,   0,   0],
#        [  8,  46,   4,  15,  12,  10, 150,   0,   6],
#        [  0,   1,   0,   0,   0,   0,   0,   0,   0],
#        [  0,   0,   0,   0,   0,   0,   0,   0,   5]])

accuracy_score(valid_preds,X_valid['Class'])
log_loss(valid_preds,X_valid['Class'])

test_preds = bst.predict(dtest)+1

#ID,class1,class2,class3,class4,class5,class6,class7,class8,class9
ans_class = np.zeros(shape=(5668,9))


j=0
for i in test_preds:
	ans_class[int(j),int(i-1)] =int(1)
	j=j+1

ans_class=ans_class.astype(int)

sub_format=pd.DataFrame(ans_class)
sub_format.columns=['class1','class2','class3','class4','class5','class6','class7','class8','class9']
sub_format['ID']=test['ID']
sub_format=sub_format[['ID','class1','class2','class3','class4','class5','class6','class7','class8','class9']]

sub_format.to_csv("submission_1st.csv",index=False)

#df_train_txt = pd.read_csv('../input/training_text', sep='\|\|', header=None, skiprows=1, names=["ID","Text"])


