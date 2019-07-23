import pandas as pd
import numpy as np
import scipy
from scipy import stats
import argparse
import sys
from my_constants import Constants


class FERPlus:
    def __init__(self, df, output_dir):
        self.df = FERPlus._preprocess(df)
        self.train = self.df[self.df['dataset'] == 'Training'].reset_index(drop=True)
        self.valid = self.df[self.df['dataset'] == 'PrivateTest'].reset_index(drop=True)
        self.test = self.df[self.df['dataset'] == 'PublicTest'].reset_index(drop=True)
        self.output_dir = output_dir

    @staticmethod
    def _calc_metrics(series):
        n_annotations = sum(series[Constants.get_emotion_cols()])
        if n_annotations == 0:
            series['entropy'] = np.nan
            series['disagreement_p'] = np.nan
            return series
        # count -> probabilities.
        probs = list(series[Constants.get_emotion_cols()]*1.0/n_annotations)
        series['entropy'] = scipy.stats.entropy(probs)
        series['disagreement_p'] = 1.0 - sum([p*p for p in probs])  # 1 - \sum p^2
        series['n_annotations'] = n_annotations
        series['emotion_corrected_label'] = Constants.correct_emotion_label(series['emotion'])
        return series

    @staticmethod
    def _preprocess(df):
        df = df.apply(FERPlus._calc_metrics, axis=1)
        df = df.dropna(subset=['img_name', 'entropy', 'disagreement_p'])
        df['emotion_corrected_label'] = df['emotion_corrected_label'].astype(int)
        df = df[['emotion', 'pixels', 'dataset', 'img_name'] + Constants.get_label_cols() +
                ['emotion_corrected_label', 'entropy', 'disagreement_p', 'n_annotations']]
        return df

    @staticmethod
    def _save_df(df, output_dir):
        df.to_csv(output_dir, index=False)

    def export_data(self):
        self._save_df(self.train, self.output_dir+'/train.csv')
        self._save_df(self.valid, self.output_dir + '/valid.csv')
        self._save_df(self.test, self.output_dir + '/test.csv')
        self._save_df(self.df, self.output_dir + '/all.csv')
        return


def main(args):
    fer = pd.read_csv(args.fer_dir)
    fer_plus = pd.read_csv(args.fer_plus_dir)
    all_df = fer.merge(fer_plus, how='inner', on='Usage', left_index=True, right_index=True)
    all_df = all_df.rename(columns={"Usage": "dataset", "Image name": "img_name"})
    fer_plus = FERPlus(all_df, args.output_dir)
    fer_plus.export_data()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--fer_dir', type=str,
                        default='/mas/u/asma_gh/uncnet/datasets/FER+/FER/fer2013/fer2013.csv',
                        help='Path to FER data.')
    parser.add_argument('--fer_plus_dir', type=str,
                        default='/mas/u/asma_gh/uncnet/datasets/FER+/FERPlus/fer2013new.csv',
                        help='Path to FER+ data.')
    parser.add_argument('--output_dir', type=str,
                        default='/mas/u/asma_gh/uncnet/datasets/FER+',
                        help='Path to the directory to save train/valid/test subsets.')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=68)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
