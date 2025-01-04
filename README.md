# MLOPS

Este projeto tem o propósito de apresentar alguns padrões de um projeto de machine learning pronto para produção.

### Informações gerais:
- Base de dados utilizada: https://www.kaggle.com/datasets/sukhmandeepsinghbrar/housing-price-dataset
- src : exemplo de código produtivo
- assets : arquivo yaml de configuração
- models: pkl do modelo treinado
- Dockerfile : Configurado para aplicar predição
- workflow: Esteira para dockerhub
    - "dockerhub_deploy" build, execução deploy da imagem no dockerhub

O repositório possui a seguinte estrutura:

```
├───.github/workflows
    └───dockerhub_deploy.yml
├───assets
    └───config.yml
├───datasets
    └───train.csv
    └───predict.csv
├───models
    └───model.pkl
└───src
    └───core.py
    └───pipeline.py
    └───predict.py
    └───train.py
    └───utils.py
└───Dockerfile  
└───tox.ini
└───requirements.txt
└───README.md
└───.gitignore
```

### Como testar localmente:

1. Instale o tox
```
pip install tox
```
2. Execute o tox
```
python -m tox
```
