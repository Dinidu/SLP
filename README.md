Single Layer Perceptron ::

Usage ::

```
main [-h] -f FILE -lr LEARNING_RATE -lmse LMSE
```

Help ::
```
python main -h

usage: main [-h] -f FILE -lr LEARNING_RATE -lmse LMSE

Single layer perceptron

optional arguments:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  dataset file in csv
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate of the algorithm
  -lmse LMSE, --lmse LMSE
                        Lowest MSE

```
Run Example::
```
python main -f test/input.csv -lr 0.001 -lmse 0.00000001
```

