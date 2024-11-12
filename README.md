# MLOPS

Este projeto tem o propósito de apresentar alguns padrões de um projeto de machine learning pronto para produção.

### Informações gerais:
- Base de dados utilizada: https://www.kaggle.com/datasets/sukhmandeepsinghbrar/housing-price-dataset
- notebooks: "draft" do projeto
- src : exemplo de código produtivo
- assets : arquivo de configuração e pkl do modelo treinado
- Dockerfile : Configurado para aplicar predição
- workflow: Esteira para dockerhub
    - "exececution_workflow" testa a execução do script
    - "docker_workflow" deploy da imagem no dockerhub e também realiza a sua execução


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

### Como testar localmente:

1. Instale o tox
```
pip install tox
```
2. Execute o tox
```
python -m tox
```
