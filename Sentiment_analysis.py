import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


data_path=os.path.abspath('data')
train_file=os.path.join(data_path,'train.tsv')
test_file=os.path.join(data_path,'test.tsv')


train=pd.read_csv(train_file,delimiter="\t",header=0)
test=pd.read_csv(test_file,delimiter="\t",header=0)

print ("Train file shape:  ", train.shape)
print ("Test file shape:    ", test.shape)





vectorizer=CountVectorizer() #initialise Bag of words
train_count=vectorizer.fit_transform(train.Phrase)
print ("Bag of words Counts: ", train_count.shape)


tf_idf=TfidfTransformer() 
train_tf_idf=tf_idf.fit_transform(train_count)
print ("Tf-Idf : ", train_tf_idf.shape)


model=MultinomialNB()
model.fit(train_tf_idf,train.Sentiment)


test_count=vectorizer.transform(test.Phrase)
test_tf_idf=tf_idf.transform(test_count)



predicted=model.predict(test_tf_idf)


output=pd.DataFrame(data={"PhraseId":test.PhraseId,"Sentiment":predicted})
output.to_csv("Sentiment Analysis on Movie Reviews -- NaiveBayes",index=False,quoting=3)
