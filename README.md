# Background
This is a HuggingFace implementation of the Differentiable Search Index.

# Code references
* https://colab.research.google.com/drive/1RFBIkTZEqbRt0jxpTHgRudYJBZTD3Szn?usp=sharing#scrollTo=AksRL_QyyYZj
* https://github.com/ArvinZhuang/DSI-transformers/tree/main


# Papers
* [Transformer Memory as a Differentiable Search Index](https://arxiv.org/abs/2202.06991)
* [How Does Generative Retrieval Scale to Millions of Passages?](https://arxiv.org/abs/2305.11841)


# Setup instructions
**NOTE** 
* Works with `Python 3.8.10`, `Python 3.9.13`, `Python 3.9.17`
* For `Python 3.10.11` -- pip installing requirements seems to work, but get an error when trying to import `datasets`
* For `Python 3.11.1` --  pip hangs trying to install multiprocess
```
# 0) clone this repo
# 1) setup virtual env
$ python3 -m venv .venv
# 2) activate venv
$ source .venv/bin/activate
# 3) install requirements
$ pip install -r requirements.txt
# 4) Install dsi package locally
$ pip install -e .
```
