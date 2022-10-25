# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd
import hmm
import re
import json

# def read_trans_prob_file(file):
#     output_dic = {}
#     # Output1: dictionary of tag: prob of transitioning to all other tags
#     with open(file, encoding = 'utf-8') as curr:
#         for line in curr.readlines():
#             tag_ij, sep, trans_prob = line.strip().partition(':')
#             tags = re.split(tag_ij,'.+,.')
#             if tags[0] not in output_dic.keys():
#                 output_dic[tags[0]] = {}
#                 output_dic[tags[0]][tags[2].strip()] = float(trans_prob.strip())
#             else:
#                 output_dic[tags[0]][tags[2].strip()] = float(trans_prob.strip())
#     return output_dic

def read_trans_prob_file(file):
    output_dic = {}
    # Output1: dictionary of tag: prob of transitioning to all other tags
    with open(file, encoding = 'utf-8') as curr:
        for line in curr.readlines():
            tag_ij, sep, trans_prob = line.strip().partition(':')
            tag_i, sep, tag_j = tag_ij.partition(' ')
            if tag_i not in output_dic.keys():
                output_dic[tag_i] = {}
                output_dic[tag_i][tag_j.strip()] = float(trans_prob.strip())
            else:
                output_dic[tag_i][tag_j.strip()] = float(trans_prob.strip())
    return output_dic

def open_file_into_dic(file):
    output_dic = {}
    token_tags = []
    output_probs_lst = []
    with open(file, encoding = 'utf-8') as curr:
        for line in curr.readlines():
            token, tag, prob = line.split(" ")
            if token not in output_dic.keys():
                output_dic[token] = {}
                output_dic[token][tag] = float(prob.replace("\n", ""))
            else:
                output_dic[token][tag] = float(prob.replace("\n", ""))
    return output_dic

def vertibipredict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):
    
    
    with open('./data/twitter_tags.txt', encoding='utf-8', newlines='\n') as tagFile:
        tags = [i[0] for i in tagFile.readlines()]
        
    
    transDist = read_trans_prob_file('./output/trans_probs.txt')
    outDist = open_file_into_dic('./output/output_probs.txt')



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    transDist = read_trans_prob_file('./output/trans_probs.txt')
    outDist = open_file_into_dic('./output/output_probs.txt')
    
    # to get all the tags
    with open('./data/twitter_tags.txt', encoding='utf-8', newline='\n') as tagFile:
        tags = [i[0] for i in tagFile.readlines()]
        
    # to get all test data
    lines =[[]]
    n=0
    train = open(r'./data/twitter_dev_no_tag.txt', encoding='utf-8', newline='\n')
    for i in train.readlines():
        if i=='\n':
            lines.append([])
            n+=1
        else:
            lines[n].append(i)
    # initialization
    
    
        
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
    # lines =[[]]
    # n=0
    # train = open(r'./data/twitter_train.txt', encoding='utf-8', newline='\n')
    # probs = pd.read_csv(r'./output/naive_output_probs.txt', index_col=0, skipfooter=25)
    # for i in train.readlines():
    #     if i=='\n':
    #         lines.append([])
    #         n+=1
    #     else:
    #         lines[n].append(i)
    # bigramDictionary = {}
    # totalTransition = {}
    # start = dict.fromkeys(probs.columns, 1)
    # outputCount = {}
    # tagCount = dict.fromkeys(probs.columns, 0)
    # delta = 1
    # for i in range(len(lines)-1):
    #     first = re.split('\t|\s{3}|\n', lines[i][0])
    #     start[first[1]] += 1
    #     for j in range(len(lines[i])-1):
    #         tPrev = re.split('\t|\s{3}|\n', lines[i][j])
    #         tagCount[tPrev[1]] += 1
    #         check = outputCount.get(tPrev[1], {}).get(tPrev[0], False)
    #         if check:
    #             outputCount[tPrev[1]][tPrev[0]] += 1
    #         elif type(outputCount.get(tPrev[1], False)) == bool:
    #             outputCount[tPrev[1]] = {tPrev[0]:1}
    #         elif type(outputCount.get(tPrev[1], {}).get(tPrev[0], False)) == bool:
    #             outputCount[tPrev[1]][tPrev[0]] = 1
    #         t = re.split('\t|\s{3}|\n', lines[i][j+1])
    #         Last = bigramDictionary.get(tPrev[0], {}).get(t[0], {}).get(tPrev[1], {}).get(t[1],False)
    #         if Last:
    #             bigramDictionary[tPrev[0]][t[0]][tPrev[1]][t[1]] += 1
    #             totalTransition[tPrev[0]] += 1
    #         elif type(bigramDictionary.get(tPrev[0], False))==bool:
    #             bigramDictionary[tPrev[0]] = {}
    #             bigramDictionary[tPrev[0]][t[0]] = {}
    #             bigramDictionary[tPrev[0]][t[0]][tPrev[1]] = {}
    #             bigramDictionary[tPrev[0]][t[0]][tPrev[1]][t[1]] = 1
    #             totalTransition[tPrev[0]] = 1
    #         elif type(bigramDictionary.get(tPrev[0], {}).get(t[0], False))==bool:
    #             bigramDictionary[tPrev[0]][t[0]] = {}
    #             bigramDictionary[tPrev[0]][t[0]][tPrev[1]] = {}
    #             bigramDictionary[tPrev[0]][t[0]][tPrev[1]][t[1]] = 1
    #         elif type(bigramDictionary.get(tPrev[0], {}).get(t[0], {}).get(tPrev[1], False))==bool:
    #             bigramDictionary[tPrev[0]][t[0]][tPrev[1]] = {}
    #             bigramDictionary[tPrev[0]][t[0]][tPrev[1]][t[1]] = 1
    #         else:
    #             bigramDictionary[tPrev[0]][t[0]][tPrev[1]][t[1]] = 1
                
    # for i in bigramDictionary:
    #     for j in bigramDictionary[i]:
    #         for k in bigramDictionary[i][j]:
    #             for l in bigramDictionary[i][j][k]:
    #                 bigramDictionary[i][j][k][l] = (bigramDictionary[i][j][k][l]+delta)/(totalTransition[i]+delta*(probs.shape[0]+1))
    
    
    # for i in outputCount:
    #     for j in outputCount[i]:
    #         outputCount[i][j] = (outputCount[i][j]+delta)/(tagCount[i]+delta*(len(totalTransition)+1))
            
    # with open('./output/trans_probs.txt', 'w', encoding='utf-8') as trans:
    #     trans.writelines(json.dumps(bigramDictionary))
    #     trans.writelines(json.dumps(start))
    # with open('./output/output_probs.txt', 'w', encoding='utf-8') as output:
    #     output.writelines(json.dumps(outputCount))

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