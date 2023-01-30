#サンプルソースの動かしかた
#前提：datasetディレクトリにtrainとtest用のMetricsデータ(CSV)を配置
cd src
pip3 install -r ../requirements.txt
#トレーニング
python3 train.py model_name=tps_2 tps_estimator=tps_estimator_2
#推論
python3 tune_knobs.py model_name=tps_2 tps_estimator=tps_estimator_2
