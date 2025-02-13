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

### GWNet(GNN based)
Done, Max node num: 6000

### AGCRN(GNN based)
Done, Max node num: 1000

### STID(MLP based)
Done, Max node num: 16326

### PatchSTG(Attention based)
doing

## Result

| **MODEL** | **MAX LINK** | **RMSE** | **MAE** | **MAPE** | **RUNTIME** |
| --- | --- | --- | --- | --- | --- |
| **DirectlyPredict** | 16326 | 3.01296 | 1.66436 | 19.4515 | - |
| **AR**(20 epoch, 90 days) | 16326 | 29.6917 | 24.2305 | 269.6297 | 17.5h |
| **MLP**(20 epoch, 90 days) | 16326 | 2.99756 | 1.70288 | 21.4932 | 10h |
| **Linear**(20 epoch, 90 days) | 16326 | 3.23547 | 2.38744 | 29.7397 | 14.4h |
| **STAEFormer**(10 epoch, 120days) | 250 | 2.51437 | 1.47355 | 17.0194 | 10h |
| **STAEFormer**(15 epoch, 90days) | 400 | 2.52072 | 1.49929 | 17.8083 | 14.5h |
| **GWNet**(20 epoch, 90 days) | 6000 | 2.33058 | 1.29555 | 15.5429 | 29h |
| **GWNet-A**(20 epoch, 90 days) | 6000 | 2.32365 | 1.29388 | 15.6817 | 27h |
| **AGCRN**(20 epoch, 90 days) | 1000 | 2.43474 | 1.40190 | 17.6032 | 16.1h |
| **STID**(20 epoch, 90 days) | 16326 | 2.53644 | 1.46874 | 18.3121 | 17.2h |
| **PatchSTG**(20 epoch, 90 days) | 16326 | 2.13185 | 1.15807 | 14.8984 | 28h |
