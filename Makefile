play:
	python src/PlayYahtzee.py

train:
	python src/YahtzeeAI.py --start-iter 0 --model-path models/yahtzee_iter_80.pth

benchmark:
	python src/YahtzeeAI.py --mode benchmark --model-path models/yahtzee_iter_2000.pth

venv:
	venv\Scripts\activate