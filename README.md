# MLOPS

Este projeto tem o propósito de apresentar alguns padrões de um projeto de machine learning pronto para produção.

### Informações gerais:
- datasets: dados para treinamento e amostra tratada para predição
- src : exemplo de código produtivo de treinamento
- assets : arquivo yaml de configuração e pkl dos modelos treinados
- Dockerfile : Configurado para aplicar treinamento
- workflow: Esteira para dockerhub
    - "dockerhub_deploy" build, execução deploy da imagem no dockerhub

O repositório possui a seguinte estrutura:

```
├───.github/workflows
    └───dockerhub_deploy.yml
├───assets
    └───config.yml
    └───models
        └───production
        └───trained
├───datasets
    └───Potencial_Novos_Clientes.txt (para treinamento)
    └───X_teste.csv (para predição)
└───src
    └───core.py
    └───train.py
    └───utils.py
└───predict.ipnyb
└───Dockerfile  
└───tox.ini
└───requirements.txt
└───README.md
└───.gitignore
```

### Como testar localmente:

1. Crie um ambiente virtual e faça a sua ativação

Mac ou Linux
```
python -m venv env 
source env/bin/activate

## For Windows
python -m venv env 
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\env\Scripts\Activate.ps1 
```
2. Instale os requirements
```
pip install -r requirements.txt
```
3. Aponte o PYTHONPATH para a raiz do projeto
```
export PYTHONPATH=$PYTHONPATH:./

## For Windows
$env:PYTHONPATH = "$env:PYTHONPATH;$(Get-Location)"
```
4. Execute a função de treinamento
```
python src/train.py 
```

OU FAÇA A EXECUÇÃO COM TOX

1. Instale o tox
```
pip install tox
```
2. Execute o tox
```
python -m tox
```

EXECUÇÃO DO CONTAINER
```
docker pull  carolzkamada/mlops:latest
docker run carolzkamada/mlops:latest 
```