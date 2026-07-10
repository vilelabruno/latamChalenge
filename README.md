## HOW TO RUN
# Coleta de dados
```
mkdir data
python3 utils/dumpOne.py cat-62
python3 utils/dumpOne.py esperas
python3 utils/dumpOne.py metaf
python3 utils/dumpOne.py metar
python3 utils/dumpOne.py satelite
python3 utils/dumpOne.py tc-prev
python3 utils/dumpOne.py tc-real
```

# Baixar imagens train
```
python3 utils/dumpImages.py
mkdir data/img
```
# Decode train imagens
```
python3 utils/decodeAllImages.py 
```
# Processa voos que passaram por foco de nuvens na train
```
python3 utils/flight_throught_area.py
```
# Realiza o processo de feature engineering para a train
```
python3 utils/feat_eng_train.py
```
# Processa voos que passaram por foco de nuvens test (faz download e verifica polígonos também)
```
mkdir data/img/test
python3 utils/flight_th_ar_test.py
```
# Realiza o processo de feature engineering para a test
```
python3 utils/feat_eng_test.py
```
# Realiza últimos tratamentos e fitting do modelo
```
python3 utils/lgb_model.py
```
