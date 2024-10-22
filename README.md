# MLOPS

Este projeto tem o propósito de apresentar alguns padrões de um projeto de machine learning pronto para produção.

### Informações gerais:
- Base de dados utilizada: https://www.kaggle.com/datasets/sukhmandeepsinghbrar/housing-price-dataset
- notebooks: "draft" do projeto
- src : exemplo de código produtivo
- assets : arquivo de configuração e pkl do modelo treinado
- Dockerfile : Configurado para aplicar predição
- workflow: Esteira para dockerhub

O repositório possui a seguinte estrutura:

```
├───.github/workflows
    └───workflow.yml
├───assets
    └───config.yml
    └───model.pkl
├───datasets
    └───train.csv
    └───predict.csv
└───notebook
    └───training_model.ipynb
└───src
    └───core.py
    └───pipeline.py
    └───predict.py
    └───train.py
    └───utils.py
└───Dockerfile
└───Makefile
└───requirements.txt
└───README.md
└───.gitignore
```

### Como utilizar:

1. Crie um ambiente virtual
```
python -m venv env
```
2. Ative o ambiente virtual
3. Instale o requirements
```
pip install -r requirements.txt
```
4. Execute o make para o processo completo
```
make all
```