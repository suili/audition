all:
	python src/extraction.py
	python src/anonymization.py
	python src/analysis.py

test:
	python src/extraction.py
	python src/anonymization.py
	python src/analysis.py --label_adjust_max_iters 0

clean:
	rm -rf analysis analysis.bak anonymization_info anonymized comments comments_paired online stats all.json anonymized.json

dep:
	python -m pip install -U -r requirements.txt
	pip install mplfonts -i https://pypi.tuna.tsinghua.edu.cn/simple
	hub install ernie_skep_sentiment_analysis==1.0.0 jieba_paddle==1.0.0

dep-user:
	python -m pip install -U --user -r requirements.txt
	pip install --user mplfonts -i https://pypi.tuna.tsinghua.edu.cn/simple
	hub install ernie_skep_sentiment_analysis==1.0.0 jieba_paddle==1.0.0

