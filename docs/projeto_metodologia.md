# Relato Metodológico do Projeto

## Objetivo e Escopo
- Problema: estimar o tempo real de voo (`target`) em segundos a partir de múltiplas fontes operacionais (trajetórias radar, meteo, regulamentos de tráfego, filas/esperas e campos derivados de imagens de tempestade).
- Unidade de predição: voo identificado por `flightid`, com duração computada como `(dt_arr - dt_dep)` no conjunto BIMTRA.
- Métrica de avaliação interna: erro quadrático médio na raiz (RMSE).

## Fontes de Dados
- **BIMTRA** (`bimtra.csv`/`bimtra2.csv`): horários de partida/chegada e identificadores de aeroporto; base para `target`, `linha = origem+destino`, `hora`, `diaSemana`.
- **CAT-62** (`cat-62.csv`): trajetórias radar com `lat`, `lon`, `flightlevel`, `speed`, `dt_radar`; sustenta agregações espaciais e temporais de voo.
- **Espera em solo** (`esperas.csv`): contagens de espera por aeroporto/hora/dia, usadas como proxy de congestão.
- **Regulação/slot** (`tc-prev.csv`, `tc-real.csv`): registros de `troca` previstos e realizados por aeroporto/hora/dia.
- **Meteorologia** (`metar.csv`, `metaf.csv` e equivalentes de teste): observações/forecasts textuais convertidas em variáveis físicas.
- **Imagens de tempestade** (`data/img/*.jpg` + `storm_areas.csv`/`test_storm_areas.csv`): polígonos extraídos por visão computacional para medir interseção de trajetórias com células convectivas.
- **Dicionários auxiliares** (`origem.csv`, `destino.csv`, `linha.csv`): mapeamento numérico de categorias fatorizadas para replicar codificação no teste.

## Preparação e Engenharia de Atributos

### Consolidação de voos
- Conversão de timestamps para `datetime` e cálculo do `target`.
- Remoção de duplicatas por `flightid` e filtragem de outliers curtos (`target > 1500` na versão principal `train_with_flight_v2_greather_1500.csv`, 286 080 linhas, 88 atributos).
- Criação de variáveis temporais (`hora`, `diaSemana`) e da chave de rota `linha`.

### Estatísticas radar (trajetória CAT-62)
- Agrupamento por `flightid` para medir:
  - **Duração do traçado radar**: `dt_radar` agregado como `(max - min).total_seconds()` → sintetizado em métricas `dt_radar_lambda*` (média, quantis e dispersões).
  - **Posição**: latitudes/longitudes inicial e final e distância Haversine estimada.
  - **Dinâmica de voo**: estatísticas de `flightlevel` (média, min, max, desvio) e `speed` (média/desvio).
- Agregação adicional por rota (`linha`) para capturar estatísticas históricas (contagem de voos, médias/STD por rota), persistidas em `resumo_voo.csv` e reintegradas aos conjuntos de treino/teste.

### Congestão e espera
- A partir de `esperas.csv`, derivação de médias e somas por aeroporto-hora, aeroporto-diaDaSemana e aeroporto global. As tabelas `esperas_hora/semana/aero.csv` são unidas por `origem`, `hora` e `diaSemana`, além da soma por `flightid` no horário exato (`HoraData`).

### Meteorologia
- Parse de METAR/METAF via biblioteca `metar.Metar` para extrair variáveis numéricas: temperatura, ponto de orvalho, velocidade/direção do vento, visibilidade, pressão (além de campos textuais de tempo/ céu).
- Para cada voo, cálculo de médias, mínimos, máximos e desvios padrão desses campos no aeroporto de origem e de destino (mesclando por `HoraData` e `HoraDataDest`), resultando em blocos de features com sufixos `_x` (origem) e `_y` (destino).

### Regulação e capacidade
- Em `tc-prev` e `tc-real`, transformação de carimbos de tempo para `hora` e `diaSemana` e agregação de `troca`/`trocareal` por aeroporto-hora, aeroporto-diaDaSemana e aeroporto global.
- Junções com a base de voos geram as features `troca_prev_*` e `troca_real*`, capturando exposição do voo a restrições previstas e efetivas.

### Polígonos de tempestade (visão computacional)
- Extração de polígonos nas imagens via OpenCV: recorte fixo, **morphological closing**, inversão/limiarização, detecção de contornos, união por proximidade (convex hull) e geração de vértices em JSON (`utils/decodeImage.py`).
- Conversão de coordenadas de pixel para lat/lon por interpolação bilinear a partir de pontos de controle conhecidos.
- União espacial com trilhas CAT-62 (hora arredondada) usando distância Haversine; contagem de interseções (`flight_through_area`) e identificação do polígono mais próximo.
- Agregação por `flightid` e junção ao treino/teste como `sum(flight_through_area)` (normalizada por 446 na inferência).

### Limpeza, codificação e padronização
- Sanitização de nomes de colunas (remoção de caracteres especiais) para compatibilidade com LightGBM.
- Codificação de `origem`, `destino` e `linha` via `pandas.factorize`, com dicionários exportados para replicação no teste.
- Padronização z-score com `StandardScaler` aplicada ao bloco de features contínuas antes do treinamento (fit no treino, transform no teste).

## Conjuntos Derivados Principais
- `data/train_with_flight_v2_greather_1500.csv`: base de treino pós-filtragem e junções (88 colunas).
- `data/test_with_flight_v2.csv` / `data/test_with_flight_v2_tcfixed.csv`: base de inferência com mesmas features reordenadas e codificadas.
- `data/test_flight_througt.csv` / `flightThourghtArea2.csv`: resultado da interseção trajeto–tempestade.

## Modelagem e Validação

### Baselines exploratórios
- **Regressão linear** após remoção iterativa de colunas com correlação |ρ|>0.9 (StandardScaler prévio) para avaliar linearidade.
- **XGBoost Regressor**: `objective=reg:squarederror`, `eta=0.15`, `max_depth=3`, `gamma=5`, treinado em `DMatrix`; avaliação hold-out com RMSE.
- **Busca de hiperparâmetros XGBoost** via `GridSearchCV` (cv=5) sobre `eta`, `max_depth`, `gamma`.
- **Stacking** com `StackingCVRegressor` combinando `RandomForestRegressor`, `XGBRegressor` e `LGBMRegressor`, com validação cruzada interna para meta-modelo.

### Pipeline principal com LightGBM
- Entrada: `X = dados.drop(["target", "flightid"])`, `y = target` padronizados.
- **Validação cruzada**: `KFold(n_splits=5, shuffle=True, random_state=42)`; métrica RMSE por partição.
- **Modelo**: `lgb.train` com `objective=regression`, `metric=rmse`, `num_leaves=31`, `learning_rate=0.05`, `feature_fraction=0.9`, `bagging_fraction=0.8`, `bagging_freq=5`, até 1000 iterações com `early_stopping(stopping_rounds=10)`.
- **Inferência por validação cruzada**: em cada dobra, predição no hold-out para RMSE e predição no teste; média aritmética das 5 saídas gera `solution`. Heurística adicional zera previsões quando `origem == destino` (`aux`).
- **Importância de atributos**: extraída via `model.feature_importance()` para análise interpretativa.

## Pipeline de Inferência
- Repetição das etapas de engenharia no notebook `responsePipe.ipynb`: derivação de `hora`/`diaSemana` a partir de `dt_dep`/`path`, merges com tabelas de espera, meteo, regulação, resumo de rota e tempestade.
- Normalização de tipos (`hora`, `diaSemana` para `int`), reordenação das colunas segundo `orderCols` do treino, eliminação de campos não utilizados (ex.: `hora_ref`, `snapshot_radar`, `metar/metaf` brutos).
- Aplicação dos mapas de fatorização e do `StandardScaler` ajustado no treino antes da predição pelo ensemble de dobras LightGBM.
- Geração de submissão final em `data/submission_flight_through_area.csv` com colunas `ID`, `solution`.

## Observações Metodológicas
- A padronização foi mantida mesmo com modelo de árvore para estabilizar escalas heterogêneas; categorias foram codificadas por inteiros (sem one-hot) para preservar cardinalidade.
- O uso de `early_stopping` com o mesmo conjunto de treino como validação pode subestimar overfitting; a métrica principal considerada é a média de RMSE das dobras.
- O módulo `sweetviz` foi empregado (`dataQuality.ipynb`) para inspeção de distribuição e possíveis vazios, mas não altera o pipeline de modelagem.
