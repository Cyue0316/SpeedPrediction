# SpeedPrediction


## Baselines

### DirectlyPredict

Done

### AutoRegression
Done, Max node num: 16326

### MLP
Done, Max node num: 16326

### STAEFromer(Transformer based)
Done, Max node num: 400

### GraphWaveNet(GNN based)
doing

## Result

| model                         | node  | RMSE    | MAE     | MAPE    | runtime |
| ----------------------------- | ----- | ------- | ------- | ------- | ----------------------------- |
| **DirectlyPredict**           | 16326 | 3.01296 | 1.66436 | 19.4515 | - |
| **STAEFormer**(10 epoch, 120days) | 250   | 2.51437 | 1.47355 | 17.0194 | 9h |
| **STAEFormer**(15 epoch, 90days) | 400   | 2.52072 | 1.49929 | 17.8083 | 14.5h |
| **MLP**(20 epoch, 90 days)-stop at 13th epoch | 16326 | 2.99756 | 1.70288 | 21.4932 | 10h |
| **Linear**(20 epoch, 90 days) | 16326 | 3.23547 | 2.38744 | 29.7397 | 14.4h |
| **AR**(20 epoch, 90 days)      | 16326 | 29.6917 | 24.2305 | 269.6297 | 17.5h |