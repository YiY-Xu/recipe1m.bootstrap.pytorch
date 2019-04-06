python -m bootstrap.run \
-o logs/recipe1m/adamine/options.yaml \
--exp.resume best_eval_epoch.metric.recall_at_1_im2recipe_mean \
--dataset.train_split \
--dataset.eval_split test