import pandas as pd
from scipy.stats import spearmanr, pearsonr

from glob import glob

def load_model_predictions(path = '/mnt/fast/nobackup/users/yl02706/iconicity/model_outputs'):
    files = glob(f'{path}/*.csv')
    files = sorted(files)
    scores = {}
    for file in files:
        model_name = file.split('/')[-1].split('.')[0]
        df = pd.read_csv(file)
        scores[model_name] = df['score'].tolist()
    return scores

def gpt_scores_and_correlation():
    df = pd.read_csv('winter.csv')
    gpt_35_df = pd.read_csv('gpt_3_5_iconicity_NEW.csv')
    gpt_4_df = pd.read_csv('gpt_4_repl_iconicity.csv', names=['index', 'word', 'rating'])

    gpt_35_scores = gpt_35_df['rating'].tolist()
    gpt_35_gold = df.iloc[gpt_35_df['index'].tolist()]['rating'].tolist()

    gpt_4_scores = gpt_4_df['rating'].tolist()
    gpt_4_gold = df.iloc[gpt_4_df['index'].tolist()]['rating'].tolist()

    print('GPT-3-5')
    print('Spearman:', spearmanr(gpt_35_scores, gpt_35_gold))
    print('Pearson:', pearsonr(gpt_35_scores, gpt_35_gold))
    print('=' * 20)
    print('GPT-4')
    print('Spearman:', spearmanr(gpt_4_scores, gpt_4_gold))
    print('Pearson:', pearsonr(gpt_4_scores, gpt_4_gold))

if __name__ == '__main__':
    correlations = {'spearman': spearmanr, 'pearson': pearsonr}

    scores = load_model_predictions()
    df = pd.read_csv('/mnt/fast/nobackup/users/yl02706/iconicity/winter.csv')
    gold_scores = df['rating'].tolist()

    for model_name, model_scores in scores.items():
        print('=' * 20)
        print(model_name)
        for correlation_name, correlation in correlations.items():
            correlation, p_value = correlation(gold_scores, model_scores)
            print('-' * 8)
            print(f'{correlation_name}:', correlation)
            print(f'{correlation_name} P-value:', p_value)