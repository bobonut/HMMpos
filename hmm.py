# Implement the six functions below
import numpy as np
import pandas as pd
import os
import re

def naiveTrain():
    tags = np.loadtxt("./data/twitter_tags.txt", dtype=object)
    df = pd.read_csv("./data/twitter_train.txt", sep="\t", names=["word", "tag"], dtype=str)
    df["Count"] = 1
    pivot = pd.pivot_table(df, values="Count", index="word", columns="tag", aggfunc=np.sum).fillna(0)
    numTag = pivot.sum(axis="rows")
    probs = pivot.div(numTag)
    probs.to_csv(r'./output/naive_output_probs.txt')
    totaTags = numTag.tolist()
    totaTags = [str(i) for i in totaTags]
    with open(r'./output/naive_output_probs.txt', 'a') as f:
        f.write('\n'.join(totaTags))
    return r'./output/naive_output_probs.txt'
    

def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    dataDir = './data/'
    outputDir = './output/'
    probs = pd.read_csv(outputDir+in_output_probs_filename, index_col=0, skipfooter=25)
    tagCount = pd.read_csv(outputDir+in_output_probs_filename, skiprows=probs.shape[0]+1, sep='\n', names=['count'])
    delta = 1
    unseenProbs = [delta/(float(i)+delta*(probs.shape[0]+1)) for i in tagCount['count']]
    tags = probs.columns.to_list()
    unseenTag = tags[np.argmax(unseenProbs)]
    test = open(dataDir+in_test_filename, encoding='utf-8', newline='\n')
    with open(outputDir+out_prediction_filename, 'w', encoding='utf-8') as output:
            for i in test.readlines():
                if i =='\n':
                    output.write('\n')
                else:
                    try:
                        output.write(tags[probs.loc[i[:-2]].argmax()]+'\n')
                        continue
                    except KeyError:
                        output.write(unseenTag+'\n')
                        continue

def naive_predict2(in_output_probs_filename, in_train_filename, in_test_filename, out_prediction_filename):
    #use Bayes' Theorem to invert the conditional probability
    #calculate p(y=j)
    #calculate p(x=w)
    dataDir = './data/'
    outputDir = './output/'
    probs = pd.read_csv(outputDir+in_output_probs_filename, index_col=0, skipfooter=25)
    tagCount = pd.read_csv(outputDir+in_output_probs_filename, skiprows=probs.shape[0]+1, sep='\n', names=['count'])
    tagTotal = sum(tagCount['count'])
    tagProb = [i/tagTotal for i in tagCount['count']]

    df = pd.read_csv(dataDir+in_train_filename, sep="\t", names=["word", "tag"], dtype=str)
    df["Count"] = 1
    pivot = pd.pivot_table(df, values="Count", index="word", columns="tag", aggfunc=np.sum).fillna(0)
    wordCount = pivot.sum(axis=1)
    wordProb = wordCount/sum(wordCount)
    invertProb = probs.mul(wordProb, axis=0).div(tagProb, axis=1)
    delta = 1
    unseenProbs = [delta/(float(i)+delta*(probs.shape[0]+1)) for i in tagCount['count']]
    tags = probs.columns.to_list()
    unseenTag = tags[np.argmax(unseenProbs)]
    test = open(dataDir+in_test_filename, encoding='utf-8', newline='\n')
    with open(outputDir+out_prediction_filename, 'w', encoding='utf-8') as output:
            for i in test.readlines():
                if i =='\n':
                    output.write('\n')
                else:
                # print(tags[probs.loc[i].argmax()]+'\n')
                    try:
                        output.write(tags[invertProb.loc[i[:-2]].argmax()]+'\n')
                        continue
                    except KeyError:
                        output.write(unseenTag+'\n')
                        continue

def trainDistributions():
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
    outputCount = {}
    tagCount = dict.fromkeys(probs.columns, 0)
    delta = 1
    for i in range(len(lines)-1):
        first = re.split('\t|\s{3}|\n', lines[i][0])
        start[first[1]] += 1
        for j in range(len(lines[i])-1):
            tPrev = re.split('\t|\s{3}|\n', lines[i][j])
            tagCount[tPrev[1]] += 1
            check = outputCount.get(tPrev[1], {}).get(tPrev[0], False)
            if check:
                outputCount[tPrev[1]][tPrev[0]] += 1
            elif type(outputCount.get(tPrev[1], False)) == bool:
                outputCount[tPrev[1]] = {tPrev[0]:1}
            elif type(outputCount.get(tPrev[1], {}).get(tPrev[0], False)) == bool:
                outputCount[tPrev[1]][tPrev[0]] = 1
            t = re.split('\t|\s{3}|\n', lines[i][j+1])
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
    
    
    for i in outputCount:
        for j in outputCount[i]:
            outputCount[i][j] = (outputCount[i][j]+delta)/(tagCount[i]+delta*(len(totalTransition)+1))
            
    with open('./output/trans_probs.txt', 'w', encoding='utf-8') as trans:
        trans.writelines(json.dumps(bigramDictionary))
        trans.writelines(json.dumps(start))
    with open('./output/output_probs.txt', 'w', encoding='utf-8') as output:
        output.writelines(json.dumps(outputCount))
        
def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):
    pass

def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                     out_predictions_filename):
    pass

def forward_backward(in_train_filename, in_tag_filename, out_trans_filename, out_output_filename,
                     max_iter, seed, thresh):
    pass

def cat_predict(in_test_filename, in_trans_probs_filename, in_output_probs_filename, in_states_filename,
                out_predictions_file):
    pass


def evaluate(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    correct = 0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        if pred == truth: correct += 1
    return correct, len(predicted_tags), correct/len(predicted_tags)

def evaluate_ave_squared_error(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    error = 0.0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        error += (int(pred) - int(truth))**2
    return error/len(predicted_tags), error, len(predicted_tags)

def run():
    '''
    You should not have to change the code in this method. We will use it to execute and evaluate your code.
    You can of course comment out the parts that are not relevant to the task that you are working on, but make sure to
    uncomment them later.
    This sequence of code corresponds to the sequence of questions in your project handout.
    '''

    ddir = '' #your working dir

    in_train_filename = f'{ddir}/twitter_train.txt'

    naive_output_probs_filename = f'{ddir}/naive_output_probs.txt'

    in_test_filename = f'{ddir}/twitter_dev_no_tag.txt'
    in_ans_filename  = f'{ddir}/twitter_dev_ans.txt'
    naive_prediction_filename = f'{ddir}/naive_predictions.txt'
    naive_predict(naive_output_probs_filename, in_test_filename, naive_prediction_filename)
    correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    print(f'Naive prediction accuracy:     {correct}/{total} = {acc}')

    naive_prediction_filename2 = f'{ddir}/naive_predictions2.txt'
    naive_predict2(naive_output_probs_filename, in_train_filename, in_test_filename, naive_prediction_filename2)
    correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')

    trans_probs_filename =  f'{ddir}/trans_probs.txt'
    output_probs_filename = f'{ddir}/output_probs.txt'

    in_tags_filename = f'{ddir}/twitter_tags.txt'
    viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
    viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
                    viterbi_predictions_filename)
    correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

    trans_probs_filename2 =  f'{ddir}/trans_probs2.txt'
    output_probs_filename2 = f'{ddir}/output_probs2.txt'

    viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
                     viterbi_predictions_filename2)
    correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}')

    in_train_filename   = f'{ddir}/twitter_train_no_tag.txt'
    in_tag_filename     = f'{ddir}/twitter_tags.txt'
    out_trans_filename  = f'{ddir}/trans_probs4.txt'
    out_output_filename = f'{ddir}/output_probs4.txt'
    max_iter = 10
    seed     = 8
    thresh   = 1e-4
    forward_backward(in_train_filename, in_tag_filename, out_trans_filename, out_output_filename,
                     max_iter, seed, thresh)

    trans_probs_filename3 =  f'{ddir}/trans_probs3.txt'
    output_probs_filename3 = f'{ddir}/output_probs3.txt'
    viterbi_predictions_filename3 = f'{ddir}/fb_predictions3.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename3, output_probs_filename3, in_test_filename,
                     viterbi_predictions_filename3)
    correct, total, acc = evaluate(viterbi_predictions_filename3, in_ans_filename)
    print(f'iter 0 prediction accuracy:    {correct}/{total} = {acc}')

    trans_probs_filename4 =  f'{ddir}/trans_probs4.txt'
    output_probs_filename4 = f'{ddir}/output_probs4.txt'
    viterbi_predictions_filename4 = f'{ddir}/fb_predictions4.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename4, output_probs_filename4, in_test_filename,
                     viterbi_predictions_filename4)
    correct, total, acc = evaluate(viterbi_predictions_filename4, in_ans_filename)
    print(f'iter 10 prediction accuracy:   {correct}/{total} = {acc}')

    in_train_filename   = f'{ddir}/cat_price_changes_train.txt'
    in_tag_filename     = f'{ddir}/cat_states.txt'
    out_trans_filename  = f'{ddir}/cat_trans_probs.txt'
    out_output_filename = f'{ddir}/cat_output_probs.txt'
    max_iter = 1000000
    seed     = 8
    thresh   = 1e-4
    forward_backward(in_train_filename, in_tag_filename, out_trans_filename, out_output_filename,
                     max_iter, seed, thresh)

    in_test_filename         = f'{ddir}/cat_price_changes_dev.txt'
    in_trans_probs_filename  = f'{ddir}/cat_trans_probs.txt'
    in_output_probs_filename = f'{ddir}/cat_output_probs.txt'
    in_states_filename       = f'{ddir}/cat_states.txt'
    predictions_filename     = f'{ddir}/cat_predictions.txt'
    cat_predict(in_test_filename, in_trans_probs_filename, in_output_probs_filename, in_states_filename,
                predictions_filename)

    in_ans_filename     = f'{ddir}/cat_price_changes_dev_ans.txt'
    ave_sq_err, sq_err, num_ex = evaluate_ave_squared_error(predictions_filename, in_ans_filename)
    print(f'average squared error for {num_ex} examples: {ave_sq_err}')

def main():
    naive_predict('naive_output_probs.txt', 'twitter_dev_no_tag.txt', 'naive_predictions.txt')
    naive_predict2('naive_output_probs.txt', 'twitter_train.txt', 'twitter_dev_no_tag.txt', 'naive_predictions2.txt' )
    print(evaluate('./output/naive_predictions.txt', './data/twitter_dev_ans.txt'))
    print(evaluate('./output/naive_predictions2.txt', './data/twitter_dev_ans.txt'))
    
if __name__ == '__main__':
    main()
