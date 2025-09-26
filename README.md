# Robust Minimax Boosting with Performance Guarantees (RMBoost)

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](/AMRC_Python) [![Made with!](https://img.shields.io/badge/Made%20with-MATLAB-red)](/AMRC_Matlab)  [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](#support-and-author)

This repository is the official implementation of Robust Minimax Boosting with Performance Guarantees

RMBoost methods are robust to general types of label noise and can also achieve strong classification performance.

## Source code

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](CL-MRC_Python) 
[![Made with!](https://img.shields.io/badge/Made%20with-MATLAB-red)](CL-MRC_Matlab)

mMBoost folder contains the Python and Matlab folders that include the Python and Matlab implementations, respectively.

### Python code

* run_RMBoost.py is the main file. In such file we can modify the number of rounds and the solver (linprog or mosek)
* RMBoost.py is the file that includes fit and predict functions

#### Requirements

The requirements are detailed in the requeriments.txt file. Run the following command to install the requeriments:

```setup
pip install -r requirements.txt
```

### Matlab code

* main.m is the main file. In such file we can modify the number of rounds and the solver (linprog or mosek)
* fit.m is the function that fits the model
* predict_boost.m is the function that obtains the predictions

## Installation and evaluation

To train and evaluate the model in the paper, run this command for Python:

```console
python run_RMboost.py

```

and for Matlab:

```console
matlab RMBoost.m
```
## Support and Author

## License 

RMBoost carries a MIT license.

## Citation

If you find useful the code in your research, please include explicit mention of our work in your publication with the following corresponding entry in your bibliography:
