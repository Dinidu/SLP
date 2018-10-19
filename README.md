Single Layer Perceptron ::

Usage ::

```
main [-h] -f FILE -lr LEARNING_RATE -ep EPOCH_VALUE
```

Help ::
```
python main -h

usage: main [-h] -f FILE -lr LEARNING_RATE -ep EPOCH_VALUE

Single layer perceptron

optional arguments:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  dataset file in csv
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate of the algorithm
  -ep EPOCH_VALUE --epochs 

```
Run Example::
```
python main -f test/input.csv -lr 0.001 -ep 5000
```

