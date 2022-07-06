# Audition Dataset Analyzer

## 安装需要的包
- python版本要求：python 3 （推荐3.9，或许更低也可以，没有测试）
- 一键安装
  - 如果是个人的anaconda环境： `make dep`
  - 如果是系统自带的python：`make dep-user`
- 手动安装
  ```sh
  python -m pip install -U -r requirements.txt    # 如果是系统的python 可能需要加--user选项以避免影响其他用户
  pip install mplfonts -i https://pypi.tuna.tsinghua.edu.cn/simple
  hub install ernie_skep_sentiment_analysis==1.0.0 jieba_paddle==1.0.0
  ```

## 使用

### 懒人用法：MAKEFILE
  - `make all`       执行三个步骤
  - `make test`      执行三个步骤，但最后不调整散点图标签位置（节约大量时间）
  - `make clean`     清理掉之前的所有中间结果（在修改生成中间步骤的代码后、或增加新的数据后需要执行）

### 正常用法
安装好需要的包之后，依次执行相应的步骤即可
```sh
python src/extraction.py        # 解析数据 
python src/anonymization.py     # 匿名化
python src/analysis.py          # 分析
```

其中可加些可选参数，但主要改改输出路径之类的，不重要，例如输入`python src/analysis.py -h`可看到如下帮助信息
<pre>
    usage: analysis.py [-h] [--k K] [--json_path JSON_PATH]
                     [--comments_path COMMENTS_PATH]
                     [--comments_paired_path COMMENTS_PAIRED_PATH]
                     [--comments_paired_plot_data COMMENTS_PAIRED_PLOT_DATA]
                     [--analysis_path ANALYSIS_PATH]
                       [--analysis_plot_data ANALYSIS_PLOT_DATA]
                       [--label_adjust_max_iters LABEL_ADJUST_MAX_ITERS]
                       [--regenerate_data]
    
    optional arguments:
      -h, --help            show this help message and exit
      --k K                 number of K-Means clusters for sentence clustering
                            (default: 30)
      --json_path JSON_PATH
                            (input) json-lines file (default: all.json)
      --comments_path COMMENTS_PATH
                            (output) path of aggregated comments of each lecturer
                            (default: comments/)
      --comments_paired_path COMMENTS_PAIRED_PATH
                            (output) path of aggregated comments of each repeated
                            lecturer-auditor pair (default: comments_paired/)
      --comments_paired_plot_data COMMENTS_PAIRED_PLOT_DATA
                            (output) plot data file for 1st-2nd time auditing same
                            course (default: comments_paired/comparison.tsv)
      --analysis_path ANALYSIS_PATH
                            (output) path for text analysis (default: analysis/)
      --analysis_plot_data ANALYSIS_PLOT_DATA
                            (output) path for scatter plot data (default:
                            analysis/plot_data.tsv)
      --label_adjust_max_iters LABEL_ADJUST_MAX_ITERS
                            maximum of iterations for label adjustement (set to 0
                            to turn off) (default: 1000)
      --regenerate_data     force regenerate plot data (default: False)
</pre> 

## 目录结构
### 代码文件：
- src
  - form.py           存储听课记录的数据结构，可通过解析xlsx得到（主要进行数据清理），处理完后可转为json格式字符串方便后续读写
  - text_analyzer.py  负责文本分析的类，包括向量表示、聚类、情感分析
  - extraction.py     第一步：解析数据：生成解析后的数据集all.json；统计表格每个域的数据存入 stats/ 目录（可检查解析的问题）；线上、线下统计online/目录
  - anonymization.py  第二步：匿名化：生成匿名化后的表格数据集anonymized/ 目录，以及anonymized.json，对应的密码表存储于anonymization_info/ 目录
  - analysis.py       第三步：数据分析，主要做两件事
    - 1. 同一门课多次被听的数据comments/ 目录，以及同一人多次听一门课的comments_paired/目录（其中有comparison.tsv及两幅“第一次-第二次”对比的bin图）
    - 2. 听课记录的一般性统计及散点图analysis/ 目录。其中包括plot_data.tsv是所有统计数据，后续png格式若干散点图，以及txt格式若干按每个属性大小排序后的结果，方便查看。sentences.txt是切句子的结果（按规则切小巨可能不准，根据这里调规则），后续clustering聚类结果及模型。

### 输入文件：
- original/                听课记录xlsx表格。注意：有些后缀xls的要手工改掉
- misc/hit_stopwords.txt   哈工大停用词表

### 输出文件：
<pre>
├── all.json                                          解析完成的表格，每行一个json，可每行分别用Form类读取。（严格说后缀不应用json，但方便chrome浏览器查看）
├── analysis                                          数据分析，包括切句子并聚类，以及散点图及其数据分析，由analysis.py生成
│   ├── plot_data.tsv                                 散点图的输入数据，'\t'分隔的csv格式，可用pandas读取
│   ├── plot_data.tsv_auditor_cx-cy.png               （多个）散点图。文件名中auditor表示每个点是一个“听课人”，cx和xy分别是两个特征的名称。
│   ├── plot_data.tsv_auditor_cx-cy_no_label.png      （多个）以no_label结尾的是没有标签的散点图。
│   ├── plot_data_auditor_sorted_by_auditor.txt       （多个）由plot_data.tsv（按auditor或lecturer聚集后）按其中某一列由大到小排序的结果。主要方便看数据
│   ├── sentences.txt                                 切句子结果（debug用）
│   ├── sentences.txt_clustering_30.txt               切好的小句聚类结果（每类内部按情感极性排序）
│   ├── sentences.txt_clustering_30_model.pkl         聚类模型（第二次运行省时间）
│   └── sentences.txt_clustering_30_names.txt         聚类标签
├── anonymization_info                                匿名化的人名和id的对应表
│   ├── school_ids.txt                                
│   └── teacher_ids.txt                               
├── anonymized                                        匿名化后的表格
│   └── 20220609-27002-25002.xlsx
├── anonymized.json                                   每行一个匿名化后的json数据
├── comments                                          每个老师所收到的评价分别整理为一个文件（评价包括 多个“教学环节亮点/建议” + “综合观察”）
│   ├── 10_俞勇.txt                                   （多个）文件名前数字为这个老师的课“被听的次数”
├── comments_paired                                   同一个老师听同一门课至少两次，对应的评价整理
│   ├── 2\_于双元\_张莉.txt                           （多个） “次数”\_授课人\_听课人
│   ├── comparison.tsv                                画图数据
│   ├── comparison.tsv_len.png                        两次听课评价长度对比（10个bin，每个的均值标准差）
│   └── comparison.tsv_score.png                      两次听课评价情感极性对比（10个bin，每个的均值标准差）
├── online                                            根据特定关键词判定的线上/线下结果聚合整理（该结果也体现在all.json中的online/offline变量）
│   ├── both.txt                                      既包含“线上”的关键词，也包含“线下”关键词的听课记录
│   ├── offline.txt
│   ├── online.txt
│   └── unknown.txt
└── stats                                             表格的每个域被填写的所有值聚合（debug、看数据用）
    ├── auditor.txt                                   这里的人名是已经经过程序清洗的，可以看清洗的质量
    ├── comment.txt
    ├── course.txt
    ├── date.txt
    ├── institution.txt
    ├── lecturer.txt
    ├── student_present.txt
    ├── student_total.txt
    ├── sub_highlight.txt
    ├── sub_length.txt
    ├── sub_note.txt
    ├── sub_step.txt
    ├── sub_student.txt
    ├── sub_suggestion.txt
    ├── sub_teacher.txt
    ├── time.txt
    └── topic.txt
</pre>

## 用到的包
-文本向量表示
  - [text2vec](https://github.com/shibing624/text2vec)
- 情感分析
  - [百度Ernie](https://github.com/PaddlePaddle/PaddleHub/tree/release/v2.2/modules/text/sentiment_analysis/ernie_skep_sentiment_analysis)
- 切词
  - [结巴分词paddle版](https://www.paddlepaddle.org.cn/hubdetail?name=jieba_paddle&en_category=LexicalAnalysis)
- 绘图
  - [seaborn](https://seaborn.pydata.org/index.html)
  - [matplotlib](https://matplotlib.org/)
  - [mplfonts](https://pypi.org/project/mplfonts/)    用于安装中文字体（解决乱码问题的一步）
  - [adjustText](https://github.com/Phlya/adjustText) 用于给散点图的每个点不重叠的文本标签
- 进度条
  - [tqdm](https://github.com/tqdm/tqdm)
- Machine Learning
  - pandas
  - sklearn
  - numpy
  - scipy

## 其他
- 标签调整很慢，调试时可加参数`python analysis.py --label_adjust_max_iters 0`来避免耗费过多时间
- 关于表格不同区域文本加权处理的问题（之前说下面的综合观察与上面教学环节七三开），目前只在统计“意见建议长度”时做了加权，其他地方没有加权，主要为了方便
