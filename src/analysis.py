import argparse
import os
from datetime import datetime
import shutil
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from adjustText import adjust_text
from scipy.stats import entropy
from tqdm import tqdm
from mplfonts.bin.cli import init
init()

from text_analyzer import TextAnalyzer
from form import Form

# deal with some issues of matplotlib
matplotlib.use('agg')
# for Chinese display. it seems these fixes do not work on axis labels
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--k', type=int, default=30, help='number of K-Means clusters for sentence clustering')

    parser.add_argument('--json_path', type=str, default='all.json', help='(input) json-lines file') # 'anonymized.json'

    parser.add_argument('--comments_path', type=str, default='comments/', help='(output) path of aggregated comments of each lecturer')
    parser.add_argument('--comments_paired_path', type=str, default='comments_paired/', help='(output) path of aggregated comments of each repeated lecturer-auditor pair')
    parser.add_argument('--comments_paired_plot_data', type=str, default='comments_paired/comparison.tsv', help='(output) plot data file for 1st-2nd time auditing same course')

    parser.add_argument('--analysis_path', type=str, default='analysis/', help='(output) path for text analysis')
    parser.add_argument('--analysis_plot_data', type=str, default='analysis/plot_data.tsv', help='(output) path for scatter plot data')
    parser.add_argument('--label_adjust_max_iters', type=int, default=1000, help='maximum of iterations for label adjustement (set to 0 to turn off)')

    parser.add_argument('--regenerate_data', action='store_true', help='force regenerate plot data')
    args = parser.parse_args()

    # STEP 1: load all the pre-parsed form data
    forms = []
    print(datetime.now(), f"reading forms from {args.json_path}")
    with open(args.json_path, 'r') as f:
        for line in f:
            forms.append(Form.from_json_str(line.strip()))

    # STEP 2: combine all remarks (comments + [highlights + suggestions] from each teaching steps) and do clustering
    print(datetime.now(), f"clustering analysis")
    os.makedirs(args.analysis_path, exist_ok=True) 
    all_remarks = ''.join(form.all_remarks() for form in forms)
    analyzer = TextAnalyzer(all_remarks, args.analysis_path, args.k)
        
    # STEP 3: group comments by lecturer, and compare multiple forms filled in by same (lecturer, auditor) pairs
    # generate comments_paired/comparison.tsv if not exist
    if not os.path.exists(args.comments_paired_plot_data) or args.regenerate_data:
        if os.path.exists(args.comments_path):
            shutil.rmtree(args.comments_path)
        os.makedirs(args.comments_path, exist_ok=True) 
        if os.path.exists(args.comments_paired_path):
            shutil.rmtree(args.comments_paired_path)
        os.makedirs(args.comments_paired_path, exist_ok=True) 
        print(datetime.now(), f"aggregating comments")
        comment_stat = defaultdict(lambda: defaultdict(dict))
        comment_pair_stat = defaultdict(lambda: defaultdict(list))
        for form in tqdm(forms):
            comment_stat[form.lecturer][(form.date, form.auditor)] = form.all_remarks()
            comment_pair_stat[form.lecturer][form.auditor].append((form.date, form.all_remarks()))
        print(datetime.now(), f"writing to {args.comments_path}")
        comparison_data = defaultdict(list)
        for lecturer in comment_stat.keys():
            date_comments = comment_stat[lecturer]
            with open(args.comments_path + f"{len(date_comments)}_{lecturer}.txt", 'w') as f:
                for (date, auditor), comment in date_comments.items():
                    f.write(f"{date}\t{auditor}\t{comment}\n")
            for (auditor, auditions) in comment_pair_stat[lecturer].items():
                if len(auditions) >= 2:
                    with open(args.comments_paired_path + f"{len(auditions)}_{lecturer}_{auditor}.txt", 'w') as f:
                        score1 = None
                        score2 = None
                        for date, comment in auditions:
                            sentiment = analyzer.get_sentiment_score(comment)
                            f.write(f"{date}\t{sentiment}\t{comment}\n")
                            if score1 is None:
                                score1 = sentiment
                                len1 = len(comment)
                            elif score2 is None:
                                score2 = sentiment
                                len2 = len(comment)
                        if score1 > 0 and score2 > 0:
                            comparison_data['score1'].append(score1)
                            comparison_data['score2'].append(score2)
                            comparison_data['len1'].append(len1)
                            comparison_data['len2'].append(len2)
        df = pd.DataFrame(comparison_data)
        df.to_csv(args.comments_paired_plot_data, sep="\t", index=False)
        print(datetime.now(), f"comments comparison plot data written to {args.comments_paired_plot_data}")

    print(datetime.now(), f"plotting comments pair comparison")
    var_names = {
        "score": "情感极性",
        "len": "文本长度",
    }
    df = pd.read_csv(args.comments_paired_plot_data, sep='\t')
    for var in ['score', 'len']:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
        sns.regplot(data=df, x =f'{var}1', y=f'{var}2', x_bins=10, fit_reg=False)
        value_range = [0.7,1] if var =='score' else [50,700]
        plt.plot(value_range, value_range, color='silver', linestyle='dashed', linewidth=2)
        plt.text(*value_range, f"{var_names[var]}“第一次听课 > 第二次听课”的课程数：{sum(df[f'{var}1'] > df[f'{var}2'])}/{len(df)}", fontsize=12)
        plt.title(f"同听课人听同一门课两次听课评价 {var_names[var]}分布")
        plt.xlabel('第一次听课')
        plt.ylabel('第二次听课')
        plt.savefig(args.comments_paired_plot_data + f"_{var}.png")


    # STEP 4: write statistics of all remarks for plotting
    # generate analysis/plot_data.tsv if not exist
    if not os.path.exists(args.analysis_plot_data) or args.regenerate_data:
        print(datetime.now(), f"generating plot data")
        data = defaultdict(list)
        cluster_dists, sentiment_dists = [], []
        texts = []
        for form in tqdm(forms):
            all_remarks = form.all_remarks()

            data['auditor'].append(form.auditor)
            data['lecturer'].append(form.lecturer)
            data['text'].append(all_remarks)
            texts.append(all_remarks)

            data['length'].append(form.length - 96) # length of an empty chart is 97

            cluster_ids, sentiments, sents = analyzer.cut_and_analysis(all_remarks)
            lengths = [len(sent) for sent in sents]
            sum_lengths = sum(lengths)
            counter = Counter()
            if sum_lengths > 0:
                for cluster_id, length in zip(cluster_ids, lengths):
                    counter.update({cluster_id :length / sum_lengths})
                data['entropy'].append(entropy([counter[i] for i in range(args.k)]))
                data['sentiment_mean'].append(np.average(sentiments, weights=lengths))
                data['sentiment_var'].append(np.sqrt(np.cov(sentiments, aweights=lengths)))
            else:
                data['entropy'].append(float('nan'))
                data['sentiment_mean'].append(float('nan'))
                data['sentiment_var'].append(float('nan'))

            pos, neg = analyzer.pos_neg_stat(form)
            data['neg_len'].append(neg)
            data['neg_ratio'].append(neg / (pos + neg))

            cluster_dist, sentiment_dist = analyzer.cluster_and_sentiment_distribution(all_remarks)
            cluster_dists.append(cluster_dist)
            sentiment_dists.append(sentiment_dist)
            for i in range(args.k):
                data[f"c{i}"].append(cluster_dist[i])
            for i in range(args.k):
                data[f"s{i}"].append(sentiment_dist[i])
        
        # visualization of 30-cluster distribution, 30-cluster sentiment distribution, 976d sentence vector
        # dimentsion reduction 30d -> 2d
        cluster_xys = analyzer.visualization(cluster_dists)
        sentiment_xys = analyzer.visualization(sentiment_dists)
        vectors = analyzer.get_vectors(texts)
        vector_xys = analyzer.visualization(vectors)
        for i in tqdm(range(len(forms))):
            data['cx'].append(cluster_xys[i][0])
            data['cy'].append(cluster_xys[i][1])
            data['sx'].append(sentiment_xys[i][0])
            data['sy'].append(sentiment_xys[i][1])
            data['x'].append(vector_xys[i][0])
            data['y'].append(vector_xys[i][1])

        df = pd.DataFrame(data)

        print(datetime.now(), f"plot data written to {args.analysis_plot_data}")
        df.to_csv(args.analysis_plot_data, sep='\t', index=False)
        
    # plot axis names and obj names
    feature_names = {
        "length":"表格详实程度",
        "neg_len":"意见建议总量",
        "neg_ratio":"意见建议占比",
        "entropy":"角度丰富程度",
        "sentiment_mean":"情感极性均值",
        "sentiment_var":"情感极性标准差",
        "cx":"30类文本占比降维x",
        "cy":"30类文本占比降维y",
        "sx":"30类情感极性降维x",
        "sy":"30类情感极性降维y",
        "x":"文本向量降维x",
        "y":"文本向量降维y",
    }
    for i in range(args.k):
        feature_names[f"c{i}"]=f"第{i}类（{analyzer.cluster_names[i]}）文本占比"
        feature_names[f"s{i}"]=f"第{i}类（{analyzer.cluster_names[i]}）情感极性"
    obj_names = {
        "record":"听课记录",
        "auditor":"听课人",
        "lecturer":"授课人",
    }
    def feature_name(s):
        return feature_names[s] if s in feature_names else s
    def obj_name(s):
        return obj_names[s] if s in obj_names else s

    # load and preprocess plot data
    df = pd.read_csv(args.analysis_plot_data, sep='\t')
    df['record'] = df.apply(lambda row: f"{row.auditor}→{row.lecturer}", axis=1)
    
    # inside dfs: 'record'->df, 'auditor'->df_grouped_by_auditor, 'lecturer'->df_group_by_lecturer
    dfs = {}
    dfs['record'] = df
    # aggregate data of the same lecturer/auditor
    for key in ['auditor', 'lecturer']:
        tmp = df.groupby([key], as_index=False)
        result = tmp.mean()
        result['count'] = tmp.size()['size']
        result['text'] = tmp['text'].apply(lambda x: '    '.join(x)).reset_index()['text']
        dfs[key] = result

    # sort each df by each column and write to file
    print(datetime.now(), f"doing simple statistics")
    for obj in ['record', 'auditor', 'lecturer']:
        df = dfs[obj]
        for feature in tqdm(df.columns):
            df.sort_values(by=[feature], ascending=[False]).to_csv(args.analysis_path + f'plot_data_{obj_name(obj)}_sorted_by_{feature_name(feature)}.txt', sep='\t')

    
    # all plots by (type_of_data_point, x_feature, y_feature)
    plot_list = [ 
        ("record", "neg_len", "length"), 
        ("record", "entropy", "length"), 
        ("record", "cx", "cy"), 
        ("record", "sx", "sy"), 
        ("record", "x", "y"), 
        ("record", "sentiment_mean", "sentiment_var"), 

        ("auditor", "neg_len", "length"), 
        ("auditor", "entropy", "length"), 
        ("auditor", "cx", "cy"), 
        ("auditor", "sx", "sy"), 
        ("auditor", "x", "y"), 
        ("auditor", "sentiment_mean", "sentiment_var"), 

        ("lecturer", "cx", "cy"), 
        ("lecturer", "sx", "sy"), 
        ("lecturer", "x", "y"), 
        ("lecturer", "sentiment_mean", "sentiment_var"), 
        ("lecturer", "sentiment_mean", "length"), 
        ("lecturer", "sentiment_mean", "neg_len"), 
    ]
    # first generate the "no label" version, then the full version for reference
    for plot_labels in [False, True]:
        for obj, x_feature, y_feature in plot_list:
            df = dfs[obj]

            print(datetime.now(), f"generating plot {obj} {x_feature}-{y_feature}")
            x_size, y_size = 32, 20
            if obj == 'record':
                x_size *= 1.5
                y_size *= 1.5
            if not plot_labels:
                x_size /= 4
                y_size /= 4
            fig, ax = plt.subplots(figsize=(x_size, y_size), dpi=200)
            ax.set(xscale="symlog", yscale="symlog")

            alpha = 0.7
            size = None if obj == "record" else 'count'
            sns.scatterplot(ax=ax, x=x_feature, y=y_feature, data=df, alpha=alpha, size=size)

            # deal with axis ticks "minus sign (-)" font problem
            tick_font = font_manager.FontProperties(family='DejaVu Sans', size=7.0)
            for labelx in ax.get_xticklabels():
                labelx.set_fontproperties(tick_font)
            for labely in ax.get_yticklabels():
                labely.set_fontproperties(tick_font)
            x_name = feature_name(x_feature)
            y_name = feature_name(y_feature)
            plt.title(f"{obj_name(obj)} “{x_name}-{y_name}” 分布情况")
            plt.xlabel(x_name)
            plt.ylabel(y_name)
            if size == 'count':
                ax.legend(title='听课次数', loc='upper left')

            # add text annotations to each nodes (lecturer/auditor names)
            if plot_labels:
                texts = []
                df.apply(lambda row: texts.append(ax.text(row[x_feature]+0.002, row[y_feature], str(row[obj]))), axis=1)
                if args.label_adjust_max_iters > 0:
                    print(datetime.now(), f"adjusting labels")
                    n_iter = adjust_text(texts, only_move={'texts':'xy'}, autoalign=False, lim=args.label_adjust_max_iters)
                    print(datetime.now(), f"done after {n_iter} iterations")

            plt.savefig(args.analysis_plot_data + f"_{obj_name(obj)}_{feature_name(x_feature)}-{feature_name(y_feature)}{'' if plot_labels else '_no_label'}.png")



if __name__ == '__main__':
    main()
