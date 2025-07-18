import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    
    args = parser.parse_args()

    total_data = list()

    for fname in os.listdir(args.data_dir):
        if not fname.endswith('.csv') or fname == 'agg.csv': continue
        df = pd.read_csv(os.path.join(args.data_dir, fname))
        total_data.append(df)

    total_data = pd.concat(total_data, ignore_index=True)
    total_data.to_csv(os.path.join(args.data_dir, 'agg.csv'))

    fig, axes = plt.subplots(1, 3)
    gb_df = total_data.groupby('eps')
    agg_df = gb_df.mean()
    q25_df = gb_df.quantile(0.25)
    q75_df = gb_df.quantile(0.75)
    eps = agg_df.index
    
    axes[0].plot(eps, agg_df['ba'])
    axes[0].fill_between(eps, q25_df['ba'], q75_df['ba'], alpha=0.2)
    axes[0].set_title('BA')
    axes[0].set_ylim(0, 1)

    axes[1].plot(eps, agg_df['f1'])
    axes[1].fill_between(eps, q25_df['f1'], q75_df['f1'], alpha=0.2)
    axes[1].set_title('F1')
    axes[1].set_ylim(0, 1)

    axes[2].plot(eps, agg_df['mcc'])
    axes[2].fill_between(eps, q25_df['mcc'], q75_df['mcc'], alpha=0.2)
    axes[2].set_title('MCC')
    axes[2].set_ylim(0, 1)

    fig.set_size_inches(10, 5)
    
    path_components = os.path.normpath(args.data_dir)
    path_components = path_components.split(os.sep)
    plot_dir = os.path.join('plots', *path_components[-3:])
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, 'plot.png')
    fig.savefig(plot_path, bbox_inches='tight')

if __name__ == '__main__':
    main()