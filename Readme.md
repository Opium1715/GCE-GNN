<a name="Z8Hwj"></a>
# Notification
This is an implementation of this paper(GCE-GNN) based on Tensorflow 2.X, which contains the extra functions as below:

- Log(output figure for loss and MRR@20, P@20)
- preformance enhanced evaluation
<a name="i1yiS"></a>
## Requirements
TensorFlow 2.X (version>=2.10 is prefer)<br />Python 3.9<br />CUDA11.6 and above is prefer<br />cudnn8.7.0 and above is prefer<br />**Caution:** For who wants to run in native-Windows, TensorFlow **2.10** was the **last** TensorFlow release that supported GPU on native-Windows.
<a name="rmrbf"></a>
## Paper data and code
T-mall dataset: [https://www.dropbox.com/sh/dbzmtq4zhzbj5o9/AAAMMlmNKL-wAAYK8QWyL9MEa/Datasets?dl=0&subfolder_nav_tracking=1](https://www.dropbox.com/sh/dbzmtq4zhzbj5o9/AAAMMlmNKL-wAAYK8QWyL9MEa/Datasets?dl=0&subfolder_nav_tracking=1)
<a name="aRMxw"></a>
## [Citation](https://github.com/CRIPAC-DIG/SR-GNN/tree/e21cfa431f74c25ae6e4ae9261deefe11d1cb488#citation)
```
Citation
@inproceedings{wang2020global,
    title={Global Context Enhanced Graph Neural Networks for Session-based Recommendation},
    author={Wang, Ziyang and Wei, Wei and Cong, Gao and Li, Xiao-Li and Mao, Xian-Ling and Qiu, Minghui},
    booktitle={Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
    pages={169--178},
    year={2020}
}
```
