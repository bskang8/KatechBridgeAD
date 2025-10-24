export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
python tools/visualization/visualize.py \
	projects/configs/BridgeAD_small_stage1.py \
	--result-path work_dirs/BridgeAD_small_stage1/results.pkl