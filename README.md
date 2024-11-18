# SpeedPrediction


## Baselines

### DirectlyPredict

Done

### AutoRegression
Doing

### MLP
Done, Max node num: 16326

### STAEFromer(Transformer based)
Done, Max node num: 400

### GraphSAGE(GNN based)
Todo

## Result

| model                         | node  | RMSE    | MAE     | MAPE    |
| ----------------------------- | ----- | ------- | ------- | ------- |
| DirectlyPredict               | 16326 | 3.01296 | 1.66436 | 19.4515 |
| STAEFormer(10 epoch, 120days) | 250   | 2.51437 | 1.47355 | 17.0194 |
| STAEFormer(15 epoch, 90days)  | 400   | 2.52072 | 1.49929 | 17.8083 |
| MLP(20 epoch, 90 days)        | 16326 | 2.99756 | 1.70288 | 21.4932 |
| Linear(100 epoch, 90 days)    | 16326 |         |         |         |
| AR(20 epoch, 90days)          | 16326 |         |         |         |