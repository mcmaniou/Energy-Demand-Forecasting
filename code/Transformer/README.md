**Unofficial** implementation of "Deep Transformer Models for Time Series Forecasting: The Influenza Prevalence Case"

(https://arxiv.org/abs/2001.08317)

The code is a modified version of (https://github.com/yuyama137/influenza) git repository, to run with our Energy case Dataset.

Train transformer model using train.csv running the train_script.py file:

```bash
python train_script.py 
```

You can use the parameters of the model either by changing the default value of arguments, or by passing the argument as parameter.

The train model will be saved under a "checkout/model" folder. (each 10 epochs a model is saved)
The model file is saved as "n_epoch.model".

After train you could run the test script:

```bash
python test.py 
```
You can use the parameters of the model either by changing the default value of arguments, or by passing the argument as parameter.

**Note**: -epn argument must has value that corresponds to a saved model name in checkout folder. For example if a model : "20_epoch.model" is saved and we want to test it we set the argument -epn equal to 20. 

## Others

**NNmodel.py**: a tensorflow ANN model

**model.py**: Transformer model implementation in Pytorch  (https://github.com/yuyama137/influenza)

**util.py**: Helper functions to load data in wright form. If you want to test alternative parametrazations and change the shape of data tou can do it from utlis.py, class EnergyDataset, method "getitem"
## Reference

- [Wu, Neo, et al. "Deep transformer models for time series forecasting: The influenza prevalence case." arXiv preprint arXiv:2001.08317 (2020).](https://arxiv.org/abs/2001.08317)
