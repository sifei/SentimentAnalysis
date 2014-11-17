#!/usr/bin/env python
import string,os,csv,re
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import nltk
from nltk import bigrams
from collections import Counter
import data_io
import numpy as np
from time import sleep
import math,sys


# get positive and negative words list without score
# Those words are from BingLiu Lexicon and MPQA using unique union set
WordPos = {}
WordNeg = {}
filePos = ['Lexicon/BingLiuLexicon/positive-words.txt','Lexicon/MPQA_pos.txt']
fileNeg = ['Lexicon/BingLiuLexicon/negative-words.txt','Lexicon/MPQA_neg.txt']
for filename in filePos:
    with open(filename,'r') as ifile:
        for line in ifile:
            WordPos[line.strip()] = 1

for filename in fileNeg:
    with open(filename,'r') as ifile:
        for line in ifile:
            WordNeg[line.strip()] = 1

# English word list and stopword list
english_words = {}
stopwords = []
engfile = open("wordsEn.txt",'r')
for word in engfile:
    english_words[word.strip()] = 1
engfile.close()

stopfile = open("stop-words.txt",'r')
for word in stopfile:
    stopwords.append(word.strip())
stopfile.close()

# get emoticon dictionary
EmoticonDictionary={}
infile = open("emoticonsWithPolarity.txt",'r')
for line in infile:
    emoticon = line.split()
    for idx in range(0,len(emoticon)-2):
        EmoticonDictionary[emoticon[idx]] = emoticon[-1]
infile.close()

# get Hashtag score.
HashtagDictionary={}
hashReader = csv.reader(open('Hashtag/Hashtag.csv'))
for term, score in hashReader:
    HashtagDictionary[term] = score


# get slang word dictionary
AcronymDictionary={}
reader = csv.reader(open("acrynom.csv"))
for acronym,expansion in reader:
    AcronymDictionary[acronym] = expansion

# get exclamation list.
ExclamationDictionary={}
infile1 = open("exclamation_word.txt",'r')
for line in infile1:
    ExclamationDictionary[line.lower().strip()] = 1
infile1.close()

def NegationModify(text):
    text = text.lower()
    modifyNegation = ''
    negation_words = ['not','n\'t','never','cannot']
    for word in text.split():
        length = len(word)
        if length < 3:
            if word == 'no':
                modifyNegation += ' NOT'
            else:
                modifyNegation += ' ' + word
        else:
            if word[-3:] in negation_words or word in negation_words:
                modifyNegation += ' NOT'
            else:
                modifyNegation += ' ' + word
    return modifyNegation
# 1) change links to ||U||
# 2) change all mention (@) to ||T||
# 3) change not,no,n't,never,cannot, those negation words to NOT
# 4) minimize duplicate characters in a word.
def normalize(text):
    text = text.lower()
    modifyURL = re.sub(r'http[s]?://[^\s<>"]+|www\.[^\s<>"]+',r'||U||',text)
    modifyTarget = re.sub(r'@[\w]+',r'||T||',modifyURL)
    modifyNegation = NegationModify(modifyTarget) #re.sub(r'not|no|n\'t|never|cannot',r' NOT',modifyTarget)
    modifyRepeat = re.sub(r'(.)\1{2,}',r'\1\1\1',modifyNegation)
    modifiedText = modifyRepeat
    return modifiedText

# read the training data, split to three group (+,-,~)
def load_training(filename):
    wordList = ""
    trainReader= csv.reader(open(filename,'r'))
    positive_wordList = ""
    negative_wordList = ""
    neutral_wordList = ""
    for text, pos in trainReader:
        text = normalize(text)
        wordList += text + ' '
        if pos == '1':
            positive_wordList += text + ' '
        elif pos == '-1':
            negative_wordList += text + ' '
        else:
            neutral_wordList += text + ' '
    return wordList


# replace emoticon with word
def replaceEmoticon(text):
    tweet = ""
    for term in text.split():
        if EmoticonDictionary.has_key(term):
            tweet += EmoticonDictionary[term]+' '
        else:
            tweet += term+' '
    return tweet

def replaceAcronym_STOP(text):
    replaced = ""
    for term in text.split():
        if AcronymDictionary.has_key(term):
            term = AcronymDictionary[term]

        replaced = replaced + ' ' + term
    return replaced

# pre-processing for the tweets. normalize, replace emoticon, slang words and stopwords
def wordList_preprocessing(wordList):
    wordList = wordList.lower()
    allTweets = normalize(wordList)
    allTweets = replaceEmoticon(allTweets)
    allTweets = replaceAcronym_STOP(allTweets)
    return allTweets

def get_tokens(wordList):
    allTweets = wordList_preprocessing(wordList)
    tokens = list(set(word_tokenize(allTweets)))
    return tokens


#POS of (negative or positive (JJ,RB,VB,NN)
def feature_POSCnt(tweet, wordScore):
    JJ_count = 0
    NN_count = 0
    VB_count = 0
    RB_count = 0
    JJ_Pcount = 0
    NN_Pcount = 0
    VB_Pcount = 0
    RB_Pcount = 0
    JJ_Ncount = 0
    NN_Ncount = 0
    VB_Ncount = 0
    RB_Ncount = 0
    JJ_Score = 0.0
    RB_Score = 0.0
    VB_Score = 0.0
    NN_Score = 0.0
    words = word_tokenize(tweet)
    pos = pos_tag(words)
    for item in pos:
        if "JJ" in item[1]:
            if WordPos.has_key(item[0]):
                JJ_Pcount += 1
            if WordNeg.has_key(item[0]):
                JJ_Ncount += 1
            JJ_count += 1
            if wordScore.has_key(item[0]):
                JJ_Score += wordScore[item[0]]

        if "NN" in item[1]:
            if WordPos.has_key(item[0]):
                NN_Pcount += 1
            if WordNeg.has_key(item[0]):
                NN_Ncount += 1
            NN_count += 1
            if(wordScore.has_key(item[0])):
                NN_Score += wordScore[item[0]]

        if "VB" in item[1]:
            if WordPos.has_key(item[0]):
                VB_Pcount += 1
            if WordNeg.has_key(item[0]):
                VB_Ncount += 1
            VB_count += 1
            if(wordScore.has_key(item[0])):
                VB_Score += wordScore[item[0]]

        if "RB" in item[1]:
            if WordPos.has_key(item[0]):
                RB_Pcount += 1
            if WordNeg.has_key(item[0]):
                RB_Ncount += 1
            RB_count += 1
            if(wordScore.has_key(item[0])):
                RB_Score += wordScore[item[0]]
    return JJ_count,NN_count,VB_count,RB_count,JJ_Pcount,NN_Pcount,VB_Pcount,RB_Pcount,JJ_Ncount,NN_Ncount,VB_Ncount,RB_Ncount,JJ_Score,RB_Score,VB_Score,NN_Score

# count how many positive, negation, and negative word
# update on 6/11 from score based classfy to word based(BingLiu Lexicon).
def feature_SentimentWordCnt(text,WordScore):
    negation_count = 0
    positive_count = 0
    negative_count = 0
    for item in text.split():
        if "NOT" in item:
            negation_count += 1
        if WordPos.has_key(item):
            positive_count += 1
        if WordNeg.has_key(item):
            negative_count += 1
    return negation_count,negative_count,positive_count

# count how many EmoCnt is positive or negative
def feature_EmoCnt(text):
    pos = 0
    neg = 0
    neutral = 0
    for item in text.split():
        if "Positive" in item:
            pos += 1
        if "Negative" in item:
            neg += 1
        if "Neutral" in item:
            neutral += 1
    return pos,neg,neutral

#count number of hashtags, capitalized words, exclamation words
def feature_CntOther(text,score):
    hashPos = 0
    hashNeg = 0
    capitalized = len([term for term in text.split() if term.isupper()])
    exclamation = 0
    for item in text.split():
        if HashtagDictionary.has_key(item):
            if HashtagDictionary[item] > 0:
                hashPos += 1
            else:
                hashNeg += 1
        if ExclamationDictionary.has_key(item.lower()):
            exclamation = 1
    return hashPos, hashNeg, capitalized, exclamation


#slangs, latin alphabets, dictionary words
def feature_NonPolarOther(text):
    #remove all punctuation
    identify = string.maketrans(',.-/:','     ')
    text = text.translate(identify)
    slang_crt = 0
    non_english_cnt = 0
    for item in word_tokenize(text):
        if AcronymDictionary.has_key(item):
            slang_crt += 1
        if not english_words.has_key(item):
            non_english_cnt += 1
    return slang_crt, non_english_cnt

#word score
def getWordScore(fileList):
    WordScore = {}
    print fileList
    for filename in fileList:
        reader = csv.reader(open(filename))
        for word, score in reader:
            if WordScore.has_key(word):
                try:
                    WordScore[word] = (float(WordScore[word]) + float(score))/2.0
                except:
                    pass
                    #print score, word
            else:
                try:
                    WordScore[word] = float(score)
                except:
                    pass
                    #print (word, score)
    return WordScore

def parser_Freq(filename):
    FeatureFreqParser = {}
    trainReader = csv.reader(open(filename))
    for text,pos in trainReader:
        with open("test.txt",'w') as f:
            f.write(text)
        os.system("~/Twitter/stanford-parser/stanford-parser/lexparser.sh test.txt > parseresult.txt")
        parsefile = open("parseresult.txt",'r')
        for line in parsefile:
            parse = line.strip()
            if FeatureFreqParser.has_key(parse):
                FeatureFreqParser[parse] += 1
            else:
                FeatureFreqParser[parse] = 1
        parsefile.close()
    for key in FeatureFreqParser.keys():
        if FeatureFreqParser[key] < 3:
            FeatureFreqParser.pop(key,None)
    return FeatureFreqParser

#bigram words score
def bigram_Freq(filename,wordList):
    FeatureFreqBi = {}
    trainReader = csv.reader(open(filename))
    for text, pos in trainReader:
        wordList += text + ' '
        for item in bigrams(text.split()):
            bigram = item[0] + ' ' + item[1]
            if FeatureFreqBi.has_key(bigram):
                FeatureFreqBi[bigram] += 1
            else:
                FeatureFreqBi[bigram] = 1
    for key in FeatureFreqBi.keys():
        if FeatureFreqBi[key] < 5:
            FeatureFreqBi.pop(key,None)
    return FeatureFreqBi

def feature_BigramScore(WordScore,text,FeatureFreqBi):
    score = 0.0
    tokens = word_tokenize(text)
    for item in bigrams(tokens):
        bigram = item[0] + ' ' + item[1]
        if WordScore.has_key(bigram) and FeatureFreqBi.has_key(bigram):
            if FeatureFreqBi[bigram] > 4:
                score += float(WordScore[bigram])
    return score



##############################
# sentiStrength
##############################

def sentiStrength(tweet):
    boosterWord = {}
    emoticonLookup = {}
    emotionLookup = {}
    idiomLookup = {}
    negatingWord = {}
    questionWord = {}
    slangLookup = {}
    booster = open('sentiStrength/BoosterWordList.txt','r')
    for line in booster:
        word,value = line.split()
        boosterWord[word] = int(value)
    booster.close()
    #emoticon = open('sentiStrength/EmoticonLookupTable.txt','r')
    #for line in emoticon:
    #    word,value = line.split()
    #    emoticonLookup[word] = value
    #emoticon.close()
    emotion = csv.reader(open('sentiStrength/EmotionLookupTable.csv'))
    for word, value in emotion:
        try:
            emotionLookup[word] = int(value)
        except:
            pass
            #rint word, value

    idiom = csv.reader(open('sentiStrength/IdiomLookupTable.csv'))
    for word,value in idiom:
        idiomLookup[word] = int(value)

    negating = open('sentiStrength/NegatingWordList.txt','r')
    for line in negating:
        word,value = line.split()
        negatingWord[word] = int(value)
    negating.close()
    question = open('sentiStrength/QuestionWords.txt','r')
    for line in question:
        word,value = line.split()
        questionWord[word] = int(value)
    question.close()
    slang = csv.reader(open('sentiStrength/SlangLookupTable.csv'))
    for word, value in slang:
        slangLookup[word] = value

    positive_score = 1
    negative_score = -1
    wordList = word_tokenize(tweet)
    if questionWord.has_key(wordList[0]) and len(wordList) > 1:
        if (idiomLookup.has_key(wordList[0]+' '+wordList[1])) and len(wordList) :
            curr_score = idiomLookup[wordList[0]+' '+wordList[1]]
            if curr_score < negative_score:
                negative_score = curr_score
            if curr_score > positive_score:
                positive_score = curr_score
        elif len(wordList) > 2 and (idiomLookup.has_key(wordList[0]+' '+wordList[1]+' '+wordList[2])) :
            curr_score = idiomLookup[wordList[0]+' '+wordList[1]]+' '+wordList[2]
            if curr_score < negative_score:
                negative_score = curr_score
            if curr_score > positive_score:
                positive_score = curr_score
        return positive_score, negative_score
    else:
        positive_score = 1
        negative_score = -1
        for curr in range(0, len(wordList)):
            curr_score = 0
            if emotionLookup.has_key(wordList[curr]):
                curr_score = emotionLookup[wordList[curr]]
                prev = curr - 1
                while prev > -1:
                    if boosterWord.has_key(wordList[prev]):
                        curr_score += boosterWord[wordList[prev]]
                        prev -= 1
                    elif negatingWord.has_key(wordList[prev]):
                        curr_score *= -1
                        prev -= 1
                    else:
                        prev = -1
                if curr_score < negative_score:
                    negative_score = curr_score
                if curr_score > positive_score:
                    positive_score = curr_score

    return positive_score, negative_score



# build training data contain all features

def build_train(filename):
    print "Start"
    trainReader = csv.reader(open(filename))


    title = ['Tweet','Sentiment']
    wordList = load_training(filename)
    tokens = get_tokens(wordList)
    title += tokens
    FeatureParse = parser_Freq(filename)
    FeatureFreqBi = bigram_Freq(filename,wordList)
    title += FeatureParse.keys()
    title += FeatureFreqBi.keys()

    UnigramWordScoreFiles = ("word_score_old.csv","word_score_140_old.csv")
    BigramWordScoreFiles = ("word_score_bi_old.csv","word_score_bi_140_old.csv")
    #UnigramWordScoreFiles = ['DAL.csv']
    UnigramWordScore = getWordScore(UnigramWordScoreFiles)
    BigramWordScore = getWordScore(BigramWordScoreFiles)
    sentiFeature = ['JJ_count','NN_count','VB_count','RB_count','JJ_Pcount','NN_Pcount','VB_Pcount','RB_Pcount','JJ_Ncount','NN_Ncount','VB_Ncount','RB_Ncount','JJ_Score','RB_Score','VB_Score','NN_Score','POS_score',
                    'Negation_Count','Positive_words','Negative_words','positive-emo','negative-emo','neutral-emo','capitalized_words','exclamation_words','slang_count','non-english_words_cnt','hashPos','hashNeg','BiScore',
                    'sentiStrengthPos','sentiStrengthNeg']



    title += sentiFeature

    toCSV = [title]
    count = 1
    for tweet, sentiment in trainReader:
        row = [tweet, sentiment]
        tweet = tweet.lower()
        tweetNorm = normalize(tweet)
        tweetEco = replaceEmoticon(tweetNorm)
        replaced = replaceAcronym_STOP(tweetEco)
        #print '\n\n\n'+replaced+'\n\n\n'
        slang_crt,non_english_cnt = feature_NonPolarOther(tweetEco)

        tweetFreq = {}
        for term in word_tokenize(replaced):#.split():
            if tweetFreq.has_key(term):
                tweetFreq[term] += 1
            else:
                tweetFreq[term] = 1

        for term in tokens:
            if tweetFreq.has_key(term):

                row.append(tweetFreq[term])#*ratio)
            else:
                row.append('0')

        # parser for tweet
        tweetParser = {}
        with open("test.txt",'w') as f:
            f.write(tweet)
        os.system("~/Twitter/stanford-parser/stanford-parser/lexparser.sh test.txt > parseresult.txt")
        parsefile = open("parseresult.txt",'r')
        for line in parsefile:
            parse = re.sub(r'-[0-9]+',r'',line.strip())
            if tweetParser.has_key(parse):
                tweetParser[parse] += 1
            else:
                tweetParser[parse] = 1
        parsefile.close()
        for parse in FeatureParse.keys():
            if tweetParser.has_key(parse):
                row.append('1')
            else:
                row.append('0')
        # end of parser for tweet

        tweet_tokens = tweet.split()
        tweetBiFreq = {}
        for item in bigrams(tweet_tokens):
            term = item[0] + ' ' + item[1]
            if tweetFreq.has_key(term):
                tweetBiFreq[term] += 1
            else:
                tweetBiFreq[term] = 1
        for biTerm in FeatureFreqBi.keys():
            if tweetBiFreq.has_key(biTerm):
                row.append('1')
            else:
                row.append('0')

        JJ_count,NN_count,VB_count,RB_count,JJP,NNP,VBP,RBP,JJN,NNN,VBN,RBN,JJS,RBS,VBS,NNS = feature_POSCnt(replaced,UnigramWordScore)
        BigramScore = feature_BigramScore(BigramWordScore,tweetNorm,FeatureFreqBi)
        POS_score = JJS+RBS+VBS+NNS
        negation_count, negative_count, positive_count = feature_SentimentWordCnt(replaced,UnigramWordScore)
        pos, neg, neutral = feature_EmoCnt(replaced)
        hashPos, hashNeg, UpperCase, exclamation = feature_CntOther(tweetNorm,sentiment)
        sentiScore = [JJ_count,NN_count,VB_count,RB_count,JJP,NNP,VBP,RBP,JJN,NNN,VBN,RBN,JJS,RBS,VBS,NNS,POS_score,negation_count,positive_count,negative_count,
                      pos, neg, neutral, UpperCase,exclamation, slang_crt, non_english_cnt, hashPos, hashNeg,BigramScore]

        row = row+sentiScore

        senti_Pos, senti_Neg = sentiStrength(replaced)
        sentiStrengthScore = [senti_Pos, senti_Neg]
        row += sentiStrengthScore
        toCSV.append(row)

        count += 1

    #print len(row),len(toCSV)

    with open("consolidate_features_all.csv",'wb') as fp:
        a = csv.writer(fp)
        a.writerows(toCSV)
        #print "train feature Done"
    return toCSV
build_train("consolidate/update_version.csv")

def build_test(filename):
    print "Start"
    trainReader = csv.reader(open(filename))


    title = ['Tweet','Sentiment']
    wordList = load_training("consolidate/consolidate_train_80_new.csv")
    tokens = get_tokens(wordList)
    title += tokens
    FeatureFreqBi = bigram_Freq("consolidate/consolidate_train_80_new.csv",wordList)
    FeatureParse = parser_Freq("consolidate/consolidate_train_80_new.csv")
    title += FeatureParse.keys()
    title += FeatureFreqBi.keys()

    UnigramWordScoreFiles = ("word_score_old.csv","word_score_140_old.csv")
    BigramWordScoreFiles = ("word_score_bi_old.csv","word_score_bi_140_old.csv")
    #UnigramWordScoreFiles = ['DAL.csv']
    UnigramWordScore = getWordScore(UnigramWordScoreFiles)
    BigramWordScore = getWordScore(BigramWordScoreFiles)
    sentiFeature = ['JJ_count','NN_count','VB_count','RB_count','JJ_Pcount','NN_Pcount','VB_Pcount','RB_Pcount','JJ_Ncount','NN_Ncount','VB_Ncount','RB_Ncount','JJ_Score','RB_Score','VB_Score','NN_Score','POS_score',
                    'Negation_Count','Positive_words','Negative_words','positive-emo','negative-emo','neutral-emo','capitalized_words','exclamation_words','slang_count','non-english_words_cnt','hashPos','hashNeg','BiScore',
                    'sentiStrengthPos','sentiStrengthNeg']



    title += sentiFeature

    toCSV = [title]
    count = 1
    for tweet, sentiment in trainReader:
        row = [tweet, sentiment]
        tweet = tweet.lower()
        tweetNorm = normalize(tweet)
        tweetEco = replaceEmoticon(tweetNorm)
        replaced = replaceAcronym_STOP(tweetEco)
        #print '\n\n\n'+replaced+'\n\n\n'
        slang_crt,non_english_cnt = feature_NonPolarOther(tweetEco)

        tweetFreq = {}
        for term in word_tokenize(replaced):#.split():
            if tweetFreq.has_key(term):
                tweetFreq[term] += 1
            else:
                tweetFreq[term] = 1

        for term in tokens:
            if tweetFreq.has_key(term):

                row.append(tweetFreq[term])#*ratio)
            else:
                row.append('0')
            # parser for tweet
        tweetParser = {}
        with open("test.txt",'w') as f:
            f.write(tweet)
        os.system("~/Twitter/stanford-parser/stanford-parser/lexparser.sh test.txt > parseresult.txt")
        parsefile = open("parseresult.txt",'r')
        for line in parsefile:
            parse = re.sub(r'-[0-9]+',r'',line.strip())
            if tweetParser.has_key(parse):
                tweetParser[parse] += 1
            else:
                tweetParser[parse] = 1
        parsefile.close()
        for parse in FeatureParse.keys():
            if tweetParser.has_key(parse):
                row.append('1')
            else:
                row.append('0')
            # end of parser for tweet

        tweet_tokens = tweet.split()
        tweetBiFreq = {}
        for item in bigrams(tweet_tokens):
            term = item[0] + ' ' + item[1]
            if tweetFreq.has_key(term):
                tweetBiFreq[term] += 1
            else:
                tweetBiFreq[term] = 1
        for biTerm in FeatureFreqBi.keys():
            if tweetBiFreq.has_key(biTerm):
                row.append('1')
            else:
                row.append('0')

        JJ_count,NN_count,VB_count,RB_count,JJP,NNP,VBP,RBP,JJN,NNN,VBN,RBN,JJS,RBS,VBS,NNS = feature_POSCnt(replaced,UnigramWordScore)
        BigramScore = feature_BigramScore(BigramWordScore,tweetNorm,FeatureFreqBi)
        POS_score = JJS+RBS+VBS+NNS
        negation_count, negative_count, positive_count = feature_SentimentWordCnt(replaced,UnigramWordScore)
        pos, neg, neutral = feature_EmoCnt(replaced)
        hashPos, hashNeg, UpperCase, exclamation = feature_CntOther(tweetNorm,sentiment)
        sentiScore = [JJ_count,NN_count,VB_count,RB_count,JJP,NNP,VBP,RBP,JJN,NNN,VBN,RBN,JJS,RBS,VBS,NNS,POS_score,negation_count,positive_count,negative_count,
                      pos, neg, neutral, UpperCase,exclamation, slang_crt, non_english_cnt, hashPos, hashNeg,BigramScore]

        row = row+sentiScore

        senti_Pos, senti_Neg = sentiStrength(replaced)
        sentiStrengthScore = [senti_Pos, senti_Neg]
        row += sentiStrengthScore
        toCSV.append(row)

        count += 1

    #print len(row),len(toCSV)

    with open("test_feature_builder.csv",'wb') as fp:
        a = csv.writer(fp)
        a.writerows(toCSV)
        #print "test feature Done"
    return toCSV
#build_test("consolidate/consolidate_test_20_new.csv")

def total_feature(toCSV_train,toCSV_test,index1,index2):
    if index1 == 0:
        toCSV = toCSV_test+toCSV_train[1:]
    elif index2 == 5126:
        toCSV = toCSV_train+toCSV_test[1:]
    else:
        toCSV = toCSV_train[:index1-1]+toCSV_test[1:]+toCSV_train[index1-1:]
        #print len(toCSV)
    with open("feature_set.csv",'wb') as fp:
        a = csv.writer(fp)
        a.writerows(toCSV)
        #print "feature done"

