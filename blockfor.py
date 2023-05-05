import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sksurv.datasets import load_flchain
from sksurv.ensemble import (RandomSurvivalForest,
                             RecursivePartitioningSurvivalForest,
                             GradientBoostingSurvivalAnalysis)
from sksurv.metrics import concordance_index_censored

def blockfor(X, y, blocks, block_method = "BlockForest", num_trees = 2000, mtry = None,
             nsets = 300, num_trees_pre = 1500, split_rule = "extratrees",
             always_select_block = 0):
    
    if y.ndim == 2 and y.shape[1] == 2:
        y = np.rec.fromarrays(y.T, names=["event", "time"])
        model_data = np.rec.append(y, X, axis=1)
    else:
        model_data = np.rec.append(y, X, axis=1)
    
    if np.issubdtype(model_data.dtype, np.integer):
        treetype = "Regression"
    elif np.issubdtype(model_data.dtype, np.floating):
        treetype = "Regression"
    elif np.issubdtype(model_data.dtype, np.object_) and model_data.dtype.names[0] == "event":
        treetype = "Survival"
    elif np.issubdtype(model_data.dtype, np.object_) and model_data.dtype.names[0] == "status":
        treetype = "Survival"
    else:
        raise ValueError("Unknown response type.")
    
    if blocks is None:
        raise ValueError("Argument 'blocks' must be provided.")
    
    ## Check parameters
    if mtry is None:
        if block_method in ["SplitWeights", "VarProb"]:
            mtry = sum([np.sqrt(len(block)) for block in blocks])
        else:
            mtry = [np.sqrt(len(block)) for block in blocks]
    
    ## Factors to numeric
    model_data = model_data.astype(np.float64)
    
    ## Set always.split.variables if SplitWeights/RandomBlock and always.select.block set
    if always_select_block > 0 and block_method in ["SplitWeights", "RandomBlock"]:
        always_split_variables = list(model_data.dtype.names[1:])[blocks[always_select_block - 1]]
    else:
        always_split_variables = None
    
    ## Set mtry of block to maximum if BlockVarSel and always.select.block set
    if always_select_block > 0 and block_method in ["BlockVarSel", "BlockForest"]:
        mtry[always_select_block - 1] = len(blocks[always_select_block - 1])
    
    ## Convert survival data
    if treetype == "Survival":
        model_data = np.rec.fromarrays(model_data[model_data.dtype.names[0:2]].T,
                                       names=["event", "time"]).astype(model_data.dtype)
    
    ## Create forest object
    if treetype == "Regression":
        forest = RandomForestRegressor(n_estimators=num_trees, max_features=mtry,
                                        bootstrap=True, oob_score=True, random_state=0)
    elif treetype == "Survival":
        forest = RandomSurvivalForest(n_estimators=num_trees, max_features=mtry,
                                       min_samples_leaf=10, random_state=0)
    
    forest.fit(model_data[list(model_data.dtype.names[1:])], model_data[model_data.dtype.names[0]])
    
    return forest
