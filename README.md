This repository contains the code for deploying an open-ended question answering model using huggingface transformers and integrated with Streamlit UI.

Model being used is BERT(Bidrectional Encoder Representation from Transformers). For better understanding of BERT you can refer to it's official paper:

https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/default/15848021.pdf

Model is being deployed using Streamlit UI which provides very neat and concise environment w/o going in too much of details.

Steps to run:

1. Make sure you have all the required libraries installed or run pip3 install requirements.txt
2. run: streamlit run streamlit_UI.py
3. If running for first time it will install Bert model for Q&A and will keep it in cache for subsequent runs
