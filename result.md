### MNIST
- config
```json
"name": "Mnist_LeNet",
    "n_gpu": 1,

    "arch": {
        "type": "MnistModel",
        "args": {}
    },
    "data_loader": {
        "type": "MnistDataLoader",
        "args":{
            "data_dir": "data/",
            "batch_size": 128,
            "shuffle": true,
            "validation_split": 0.1,
            "num_workers": 2
        }
    },
```
- test result  
{'loss': 0.031903635972738265, 'top_k_acc': 0.9994, 'accuracy': 0.9903}


### MIT-BIH
- introduce  
    - 48 half-hour two-lead ECG recordings obtained from 48 patients
    - digitized at 360 samples per second per channel with 11-bit resolution over a 10 mV range
    - 41 labels for annotation for different meanings
    - one recording of the dataset contains 650000 sampling spots, and two leads which are 'MLII' and 'V5' respectively

    - 标签  
    按照AAMI标准分成5大类：

        - N:0
            - N: normal beat 正常心搏
            - L: left bundle branch block beat 左束支传导阻滞心搏
            - R: right bundle branch block beat 右束支传导阻滞心搏
        - SVEB:1
            - S: premature or ectopic supraventricular escape beat 早搏或室上性异位心搏
            - A: atrial premature beat 房性早搏
            - J: nodal premature beat 交界性早搏
            - a: abberated atrial premature beat 异常房性早搏
            - e: atrial escape beat 房性逸搏
            - j: nodal escape beat 交界性逸搏
        - VEB:2
            - V: premature ventricular contraction 室性早搏
            - E: ventricular escape beat 室性逸搏
        - F:3
            - F: fusion of ventricular and normal beat 心室融合心搏
        - Q:4
            - : paced beat 起搏心拍
            - f: fusion of paced and normal beat 起搏融合心跳
            - Q: unclassified beat 未分类心拍
    - 找到R波，前溯99个采样点，后取201个采样点，约300/360=0.83秒为一个心拍
    - 因为有4段recording用了起搏器， 按照AAMI标准排除在外， 分别是102,104,107和217
- config
```json
 "name": "Mitbih_LeNet",
    "n_gpu": 1,

    "arch": {
        "type": "MitbihModel",
        "args": {
            "num_classes": 5
        }
    },
    "data_loader": {
        "type": "MitbihDataLoader",
        "args":{
            "data_dir": "./data/MIT-BIH/processedData.mat",
            "batch_size": 128,
            "shuffle": true,
            "validation_split": 0.1,
            "num_workers": 2
        }
    },

```

- train result  
    epoch          : 100  
    val_loss       : 0.019129676707819395  
    val_top_k_acc  : 1.0  
    top_k_acc      : 0.9999139592511013  
    accuracy       : 0.9921358755506607  
    val_accuracy   : 0.9925198853126156  
    loss           : 0.027109627355513723  
    
- test result  
{'loss': 0.02252106818805135, 'top_k_acc': 0.9999276855768883, 'accuracy': 0.9938532740355064}