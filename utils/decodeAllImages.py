import os
import subprocess
import pandas as pd
import json
# Cria o arquivo de log
log_file = open('log/log.txt', 'w')

# Cria o dataset Pandas

data_list = []
#dataset = pd.DataFrame(columns=['data','X','Y','poly','vertice'])

# Percorre cada arquivo na pasta 'data/img/'
for file in os.listdir('data/img/'):
    # Obtém o caminho completo do arquivo
    file_path = os.path.join('data/img/', file)
    
    # Chama o script 'utils/decodeImage.py' passando o caminho da imagem como parâmetro
    try:
        output = subprocess.check_output(['python3', 'utils/decodeImage.py', file_path])
        output = json.loads(output)
        
        # Faz o processamento necessário no 'output' para extrair as informações desejadas
        # Supondo que as informações extr
        # Supondo que as informações extraídas estejam no formato mencionado no exemplo
        # Faz o parse do output e extrai as coordenadas e polígonos
        polyCount = 0
        for poly in output:
            verticeCount = 0
            for vertice in poly:
                x = vertice[0][0]
                y = vertice[0][1]
            
            # Adiciona os dados no dataset Pandas
                data_list.append({'data': file,
                                          'X': x, 
                                          'Y': y,
                                          'poly': polyCount,
                                          'vertice': verticeCount})
                verticeCount = verticeCount + 1
            polyCount = polyCount + 1
        # Registra no arquivo de log que não ocorreu erro na execução
        log_file.write(f'{file}: sem erros\n')
        print(data_list)
    except subprocess.CalledProcessError as err:
        # Registra no arquivo de log que ocorreu um erro na execução
        log_file.write(f'{file}: erro na execução - {err}\n')
    
dataset = pd.DataFrame(data_list)
# Fecha o arquivo de log
log_file.close()

# Exibe o dataset Pandas
print(dataset)
dataset.to_csv('data/imageData.csv', index=False)
