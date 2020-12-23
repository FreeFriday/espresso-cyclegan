## Introduction
Pytorch implementation of CycleGAN based on a [repository](https://github.com/aitorzip/PyTorch-CycleGAN).

## Prerequisites
Code is tested on 
```
Python: 3.7.9
Pytorch: 1.6.0
```

## Dataset
```
├── dataset
   ├── domainA
       ├── 000001.jpg 
       ├── 000002.jpg
        ...
   └── domainB
       ├── 000001.jpg 
       ├── 000002.jpg
        ...
```

## Training
```
python train.py --path_A=dataset/domainA/* 
                --path_B=dataset/domainB/* 
                --batch_size=3 
                --name=result_dir_name
```

You can get the result at `./results/result_dir_name`.


## Test
```
python inference.py --model=snapshot/path.pt
                    --input=input/path.png 
                    --output=output/path.png 
                    --output_inout=output_with_input/path.png
```
