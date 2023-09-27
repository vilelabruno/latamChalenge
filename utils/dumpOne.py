import requests
import csv
import datetime
endarr = ["cat-62","esperas","metaf","metar","satelite","tc-prev","tc-real"]
endpoint = endarr[0]
base_url = "http://montreal.icea.decea.mil.br:5002/api/v1/"+endpoint
token = "a779d04f85c4bf6cfa586d30aaec57c44e9b7173"
REMOVER
# Definindo as datas inicial e final
start_date = datetime.date(2020, 1, 1)
start_date2 = datetime.date(2023, 1, 1)
end_date = datetime.date.today()

# Iterando a cada dois meses
delta = datetime.timedelta(days=1)
current_date = start_date
date_format = "%Y-%m-%d %H:%M:%S.001" # %Y-%m-%d

# Criando o arquivo CSV
filename = "data/"+endpoint+".csv"
csv_file = open(filename, "w", newline="")
csv_writer = csv.writer(csv_file)
params = {
    "token": token,
    "idate": start_date2.strftime(date_format),
    "fdate": (start_date2 + delta).strftime(date_format)
}
response = requests.get(base_url, params=params, headers={"accept": "application/json"})
data = response.json()
csv_writer.writerow(data[0].keys())
# Fazendo uma solicitação para cada data e armazenando os dados no CSV
while current_date <= end_date:
    
    params = {
        "token": token,
        "idate": current_date.strftime(date_format),
        "fdate": (current_date + delta).strftime(date_format)
    }
    print(params)
    response = requests.get(base_url, params=params, headers={"accept": "application/json"})
    data = response.json()

    # Verificando se há dados para armazenar
    if data:
        
        # Iterando sobre os objetos da resposta JSON
        if current_date == start_date:  # Escreve os cabeçalhos apenas na primeira iteração
            csv_writer.writerow(data[0].keys())
        for item in data:
            csv_writer.writerow(item.values())

    current_date += delta

csv_file.close()

print("Dados foram armazenados no arquivo", filename)
