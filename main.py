# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd
import hmm
import re


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # tags = np.loadtxt("./data/twitter_tags.txt", dtype=object)
    # df = pd.read_csv("./data/twitter_train.txt", sep="\t", names=["word", "tag"], dtype=str)
    # df["Count"]=1
    # pivot = pd.pivot_table(df, values="Count", index="word", columns="tag", aggfunc=np.sum).fillna(0)
    # # delta = 1
    # # pivot += delta
    # pivot.reset_index()
    # numTag = pivot.sum(axis="rows")
    # probs = pivot.div(numTag)
    # probs.to_csv(r'./output/naive_output_probs.txt')
    # print(pivot.iloc[0])
    # # with open('./data/twitter_train.txt', newline="") as twitter:
    # #     reader = csv.reader(twitter, delimiter='\t')
    # #     for i in reader:
    # #         print(i)
    
    # probs = pd.read_csv(r'./output/naive_output_probs.txt', index_col=0, skipfooter=25)
    # tagCount = pd.read_csv(r'./output/naive_output_probs.txt', skiprows=probs.shape[0]+1, sep='\n', names=['count'])
    # tagTotal = sum(tagCount['count'])
    # tagProb = [i/tagTotal for i in tagCount['count']]

    # df = pd.read_csv("./data/twitter_train.txt", sep="\t", names=["word", "tag"], dtype=str)
    # df["Count"] = 1
    # pivot = pd.pivot_table(df, values="Count", index="word", columns="tag", aggfunc=np.sum).fillna(0)
    # wordCount = pivot.sum(axis=1)
    # wordProb = wordCount/sum(wordCount)
    # invertProb = probs.mul(wordProb, axis=0).div(tagProb, axis=1)
    # delta = 1
    # unseenProbs = [delta/(float(i)+delta*(probs.shape[0]+1)) for i in tagCount['count']]
    # tags = probs.columns.to_list()
    # unseenTag = tags[np.argmax(unseenProbs)]
    # test = open(r'./data/twitter_dev_no_tag.txt', encoding='utf-8', newline='\n')
    # with open(r'./output/naive_predictions2.txt', 'w', encoding='utf-8') as output:
    #         for i in test.readlines():
    #             if i =='\n':
    #                 output.write('\n')
    #             else:
    #             # print(tags[probs.loc[i].argmax()]+'\n')
    #                 try:
    #                     output.write(tags[invertProb.loc[i[:-2]].argmax()]+'\n')
    #                     continue
    #                 except KeyError:
    #                     output.write(unseenTag+'\n')
    #                     continue
    lines =[[]]
    n=0
    train = open(r'./data/twitter_train.txt', encoding='utf-8', newline='\n')
    probs = pd.read_csv(r'./output/naive_output_probs.txt', index_col=0, skipfooter=25)
    for i in train.readlines():
        if i=='\n':
            lines.append([])
            n+=1
        else:
            lines[n].append(i)
    bigramDictionary = {}
    totalTransition = {}
    start = dict.fromkeys(probs.columns, 1)
    delta = 1
    for i in range(len(lines)-1):
        first = re.split('\t|\s{3}|\n', lines[i][0])
        # try:
        start[first[1]] += 1
        # except KeyError:
        #     start[first[1]] = 1
        for j in range(len(lines[i])-1):
            tPrev = re.split('\t|\s{3}|\n', lines[i][j])
            t = re.split('\t|\s{3}|\n', lines[i][j+1])
            # First = bigramDictionary.get(tPrev[0], False)
            # Second = 
            Last = bigramDictionary.get(tPrev[0], {}).get(t[0], {}).get(tPrev[1], {}).get(t[1],False)
            if Last:
                bigramDictionary[tPrev[0]][t[0]][tPrev[1]][t[1]] += 1
                totalTransition[tPrev[0]] += 1
            elif type(bigramDictionary.get(tPrev[0], False))==bool:
                bigramDictionary[tPrev[0]] = {}
                bigramDictionary[tPrev[0]][t[0]] = {}
                bigramDictionary[tPrev[0]][t[0]][tPrev[1]] = {}
                bigramDictionary[tPrev[0]][t[0]][tPrev[1]][t[1]] = 1
                totalTransition[tPrev[0]] = 1
            elif type(bigramDictionary.get(tPrev[0], {}).get(t[0], False))==bool:
                bigramDictionary[tPrev[0]][t[0]] = {}
                bigramDictionary[tPrev[0]][t[0]][tPrev[1]] = {}
                bigramDictionary[tPrev[0]][t[0]][tPrev[1]][t[1]] = 1
            elif type(bigramDictionary.get(tPrev[0], {}).get(t[0], {}).get(tPrev[1], False))==bool:
                bigramDictionary[tPrev[0]][t[0]][tPrev[1]] = {}
                bigramDictionary[tPrev[0]][t[0]][tPrev[1]][t[1]] = 1
            else:
                bigramDictionary[tPrev[0]][t[0]][tPrev[1]][t[1]] = 1
                
    for i in bigramDictionary:
        for j in bigramDictionary[i]:
            for k in bigramDictionary[i][j]:
                for l in bigramDictionary[i][j][k]:
                    bigramDictionary[i][j][k][l] = (bigramDictionary[i][j][k][l]+delta)/(totalTransition[i]+delta*(probs.shape[0]+1))
        
        
                
            # try:
            #     # bigramDictionary[tPrev[0]][t[0]][tPrev[1]][t[1]] += 1
            #     x = bigramDictionary.get(tPrev[0], {}).get(t[0], {}).get(tPrev[1], {}).get(t[1], 1)
            # except KeyError:
            #     # bigramDictionary[tPrev[0]][t[0]][tPrev[1]][t[1]] = 1
            
            
                                            
                
                
    # test = pd.read_csv(r'./data/twitter_dev_no_tag.txt', sep='\n\n', error_bad_lines=False,names=['word'])
    # print(type(probs.loc['!']))
    # print(probs.sum(axis=0))
    # print(sum(unseenProbs))
    
    # tags = np.loadtxt("./data/twitter_tags.txt", dtype=object)
    # df = pd.read_csv("./data/twitter_train.txt", sep="\t", names=["word", "tag"], dtype=str)
    # df["Count"] = 1
    # pivot = pd.pivot_table(df, values="Count", index="word", columns="tag", aggfunc=np.sum).fillna(0)
    # # delta = 1
    # # pivot += delta
    # # pivot.reset_index()
    # numTag = pivot.sum(axis="rows")
    # # numTag += (pivot.shape[0]+1)
    # probs = pivot.div(numTag)

    # probs.to_csv(r'./output/naive_output_probs.txt')
    # hello = numTag.tolist()
    # hello = [str(i) for i in hello]
    # with open(r'./output/naive_output_probs.txt', 'a') as f:
    #     f.write('\n'.join(hello))



# See PyCharm help at https://www.jetbrains.com/help/pycharm/