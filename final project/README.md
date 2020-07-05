# CS420-SJTU-Project

## Usage

### FCN32s

#### Training stage

```
python train_fcn32s.py 
```

#### Test stage

```
python train_fcn32s.py --mode=test --resume=True --name=fcn32s.pkl
```



### FCN16s

#### Training stage

```
python train_fcn16s.py --fcn32=fcn32s.pkl
```

#### Test stage

```
python train_fcn16s.py --mode=test --resume=True --name=fcn16s.pkl
```



### FCN8s

#### Training stage

```
python train_fcn8s.py --fcn16=fcn16s.pkl
```

#### Test stage

```
python train_fcn8s.py --mode=test --resume=True --name=fcn8s.pkl
```



### Unet

#### Training stage

```
python train.py --root='../dataset/membrane' --save_path='unet_membrane.hdf5'
```

#### Test stage

```
python test.py --root='../dataset/membrane' --model_path='unet_membrane.hdf5'
```

<<<<<<< HEAD
### Unet++

#### Training stage

```
python train.py
```

#### Test stage

```
python val.py
```
#### Calculate accuracy

```
python acc.py
```
You should place the predicted image to be tested and label in the img and label folders respectively

=======
**optional :** run ```unet_densecrf.py```to take **Unet** as the front end and **DenseCRF** as the back end
>>>>>>> 57e14b3e6cf7dfcdf97ed60f453f5e2d2d314c4f

