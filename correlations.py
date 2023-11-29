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