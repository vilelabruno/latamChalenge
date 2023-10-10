import pandas as pd
import requests

def download_images(df):
    unlock = False
    for _, row in df.iterrows():
        data = row['data']
        path = row['path']
        tamanho = row['tamanho']
        
        # Extrai o nome do arquivo da URL
        nome_arquivo = path.split('/')[-1]
        print(nome_arquivo)
        #if nome_arquivo == 'S11635384_202208250300.jpg': # Caso o nome do arquivo não seja encontrado
        #    unlock = True
        #if ~unlock:
        #    continue
        ## Define o caminho onde a imagem será salva
        pasta_destino = f'data/img/'
        
        # Faz o download da imagem usando a biblioteca requests
        response = requests.get(path)
        
        # Verifica se a resposta foi bem sucedida (código 200)
        if response.status_code == 200:
            with open(f'{pasta_destino}{data}.jpg', 'wb') as file:
                file.write(response.content)
                print(f"Imagem {nome_arquivo} salva em {pasta_destino}")
        else:
            print(f"Erro ao baixar a imagem {nome_arquivo}")

# Carrega o arquivo CSV usando pandas
dataframe = pd.read_csv('data/satelite.csv')

# Chama a função para fazer o download de cada imagem
download_images(dataframe)
