import requests
import pandas as pd
from datetime import timedelta
import sys
url = "http://montreal.icea.decea.mil.br:5002/api/v1/{0}?token=a779d04f85c4bf6cfa586d30aaec57c44e9b7173".format(sys.argv[1])
headers = {
  'accept': 'application/json'
}

# Gere todas as datas entre 2022 e 2023
dates = pd.date_range(start="2022-01-01", end="2023-12-31")

all_data = []

# Itere sobre todas as datas
for date in dates:
    # Defina o intervalo de datas para a requisição
    idate = date.strftime('%Y-%m-%d')
    fdate = (date + timedelta(days=1)).strftime('%Y-%m-%d')

    # Construa a URL final
    final_url = f"{url}&idate={idate}&fdate={fdate}"

    # Fazer a requisição
    response = requests.request("GET", final_url, headers=headers)
    
    # Verificar se a resposta é válida (status code 200)
    if response.status_code == 200:
        data = response.json()
        print(data)
        all_data.extend(data)

# Substituir a coluna 'data'
all_data_df = pd.DataFrame(all_data)

# Salvar os dados em um arquivo CSV
all_data_df.to_csv('data/{0}.csv'.format(sys.argv[1]), index=False)
