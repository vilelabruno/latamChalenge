# Versão remota "finished phase 1" (arquivada em merge de 2026-07-10)

Este diretório preserva os arquivos do branch remoto `origin/main` no ponto em
que ele divergiu da versão local que acabou virando o `main` atual do
repositório. O histórico se ramificou logo após o commit `response pipe`
(2023-10-04) e seguiu dois caminhos paralelos:

- **Caminho local (mantido nos arquivos principais do repo hoje):**
  `reg analysis and lightgbm` → `fixing` → `finalizando pipe` (out/2023) →
  `updating changes` (2026-07-10).
- **Caminho remoto (arquivado aqui):**
  `response pipe` → `401 rmse` → `finished phase 1` (até 2023-10-09,
  commit `c40eb4f`).

O caminho remoto chegou a um modelo funcional com RMSE 401 e foi marcado
como "finished phase 1", mas nunca foi incorporado ao branch local antes de
o trabalho local seguir adiante. Para não perder esse resultado, os arquivos
que entraram em conflito no merge foram salvos aqui como estavam em
`c40eb4f` (origin/main), enquanto os arquivos principais do repositório
mantiveram a versão local mais recente.

## Arquivos e o que cada um representa

| Arquivo aqui | Original (remoto) | Contrapartida local (raiz do repo) | Diferença principal |
|---|---|---|---|
| `dumpOne.py` | `utils/dumpOne.py` @ c40eb4f | `utils/dumpOne.py` | Remoto: reescrita usando `pandas`, coleta por range fixo 2022-01-01–2023-12-31, salva direto em DataFrame/CSV. Local: versão original com `csv`/loop diário desde 2020-01-01 até hoje. São duas implementações distintas do mesmo coletor, não incrementais entre si. |
| `responsePipe.ipynb` | idem @ c40eb4f | `responsePipe.ipynb` | Remoto tem 38 células, local tem 49. Pipelines de resposta divergentes — remoto é o que gerou o RMSE 401 de "finished phase 1". |
| `dataPrep.ipynb` | idem @ c40eb4f | `dataPrep.ipynb` | Notebooks de preparação de dados divergiram após o split; conteúdo e tamanho diferentes. |
| `dataQuality.ipynb` | idem @ c40eb4f | `dataQuality.ipynb` | Idem — análise de qualidade de dados evoluiu separadamente nos dois branches. |
| `lightgbmBaseline.ipynb` | idem @ c40eb4f | `lightgbmBaseline.ipynb` | Ambos os lados criaram um notebook de baseline LightGBM com o mesmo nome, mas com conteúdo distinto (add/add conflict). |
| `SWEETVIZ_REPORT.html` | idem @ c40eb4f | `SWEETVIZ_REPORT.html` | Relatório gerado automaticamente (Sweetviz) sobre datasets diferentes usados em cada branch. |

## Como recuperar o histórico completo, se precisar

O commit remoto original com todo esse trabalho continua acessível via:

```
git show c40eb4f      # "finished phase 1"
git log c40eb4f        # histórico completo do caminho remoto
```

Nada foi perdido — este diretório é só uma cópia de trabalho para consulta
rápida sem precisar navegar pelo histórico do git.
