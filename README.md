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
Done, Max node num: 16326

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


## Experiments

| MODEL                                                | MAX LINK | RMSE    | MAE     | MAPE     | RUNTIME | 单EPOCH时间 | BEST EPOCH | EPOCHS | 参数量   | 推理时间 |
| ---------------------------------------------------- | -------- | ------- | ------- | -------- | ------- | ----------- | ---------- | ------ | -------- | -------- |
| PatchSTG(20 epoch, 90 days)                          | 16326    | 2.13185 | 1.15807 | 14.8984  | 28h     | 1.5h        | 19         | 20     | 2081772  | 10.25    |
| PatchSTG+patch内动态padding                          | 16326    | 2.13987 | 1.16757 | 15.007   | 64h     | 2.8h        | 17         | 20     | 2081772  | 9.85     |
| PatchSTG+stat+广度GCN+深度2 4 cross+输入平滑         | 16326    | 2.01658 | 1.13019 | 14.4961  | 21h     | 1.5h        | 4          | 14     | 10771156 | 10.12    |
| PatchSTG+stat+广度GCN+深度cross不共享+平滑+切分修改  | 16326    |         |         |          |         |             |            |        | 10771156 |          |
| PatchSTG+stat+广度GCN+深度cross不共享+输入平滑       | 16326    |         |         |          |         |             |            |        | 10771156 |          |
| PatchSTG+stat+广度GCN+深度第2、4替换cross attn       | 16326    | 2.02566 | 1.13078 | 14.4847  | 38h     | 2.0h        | 11         | 20     | 10771156 | 12.26    |
| PatchSTG+stat+广度GCN+深度第2、4替换cross  attn+self | 16326    | 2.04227 | 1.13458 | 14.2601  | 29h     | 1.8h        | 6          | 16     | 11338996 | 10.81    |
| PatchSTG+stat+广度GCN+输入平滑                       | 16326    | 2.05476 | 1.13274 | 14.6564  | 26h     | 1.6h        | 8          | 18     | 10203316 | 9.42     |
| PatchSTG+status embdding                             | 16326    | 2.10362 | 1.14963 | 14.7278  | 32h     | 1.7h        | 20         | 20     | 2240588  | 11.12    |
| PatchSTG+status+广度attn替换为agcrn版GCN             | 16326    | 2.13135 | 1.16088 | 15.1608  | 38h     | 1.7h        | 20         | 20     |          | 9.4      |
| PatchSTG+status+广度attn替换为gwnet版GCN             | 16326    | 2.11794 | 1.15425 | 14.8269  | 36h     | 1.8h        | 20         | 20     |          | 9.74     |
| PatchSTG+status+广度attn替换为可变GCN                | 16326    | 2.02105 | 1.13254 | 14.3005  | 24h     | 1.6h        | 5          | 15     | 10203316 | 8.93     |
| PatchSTG+status+广度attn替换为可变GCN+node_emb替换   | 16326    | 2.04267 | 1.13359 | 14.6955  | 26h     | 1.6h        | 6          | 16     |          | 9.58     |
| PatchSTG+status+广度GCN+区域emb                      | 16326    | 2.03938 | 1.13895 | 14.5819  | 38h     | 2.0h        | 12         | 20     |          | 10.52    |
| PatchSTG+status+广度深度GCN                          | 16326    | 2.02571 | 1.13197 | 14.6031  | 138h    | 6.5h        | 17         | 20     | 10301724 | 21.01    |
| PatchSTG+status+广度深度GCN+Linear共享参数           | 16326    | 2.02359 | 1.12481 | 14.4386  | 91h     | 6.5h        | 3          | 13     |          | 21.05    |