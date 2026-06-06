play:
	python src/PlayYahtzee.py

train:
	python src/YahtzeeAI.py --start-iter 0 --model-path models/best_model.pth

train_ensemble:
	python src/YahtzeeAI.py --start-iter 0 --mode train_ensemble

benchmark:
	python src/YahtzeeAI.py --mode benchmark --model-path models/best_model.pth

train_categories:
	python src/YahtzeeAI.py --start-iter 0 --mode train_categories --cat-model-path models/multi_category_net.pth

venv:
	venv\Scripts\activate
