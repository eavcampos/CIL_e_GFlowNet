# Estratégia 2: pseudo-negativos no espaço do backbone congelado

Aprendizado incremental de classes (CIL) com backbone congelado e um Adapter por tarefa. A
Estratégia 2 ensina cada Adapter novo a **recusar classes antigas**, usando exemplos sintéticos
dessas classes (pseudo-negativos), sem guardar nenhuma imagem.

## O problema

Cada Adapter é treinado só com as classes da sua tarefa e nunca vê as outras. Na inferência, uma
imagem de classe antiga recebe score do Adapter certo e dos Adapters treinados depois, e às vezes
um Adapter novo ganha. Esse erro de roteamento entre tarefas é o **vazamento**:

- `vazamento = P(tarefa prevista ≠ tarefa verdadeira)`
- decomposição exata: `1 - acurácia = vazamento + erro dentro da tarefa`

**Cuidado ao interpretar.** Com tarefas formadas por classes sorteadas, mesmo erros ao acaso
caem em outra tarefa com probabilidade `(C - C_t) / (C - 1)`: com 100 classes e 20 por tarefa,
cerca de 81% dos erros contam como vazamento sem que haja nada de especial no roteamento. Por isso
o vazamento é sempre lido **junto da referência NCM** (abaixo), e o que importa é o excesso sobre ela.

## A ideia

O backbone é congelado, então `phi = backbone(imagem)` de uma classe antiga nunca muda. Ao fim de
cada tarefa, o banco guarda a média e uma covariância de posto baixo de `phi` por classe. Nas
tarefas seguintes, `phi` sorteados desse banco entram na perda com o **rótulo antigo**:

```
perda = CE_real + lambda_pn * CE_pseudo        (lambda_lvq = 0 nos configs atuais)
```

Os Adapters antigos estão congelados, então a única forma de reduzir `CE_pseudo` é **baixar o
score das classes novas** para esses `phi`.

## Os quatro configs do estudo

| Config | `pseudo_negativos.modo` | Papel |
| --- | --- | --- |
| `config_ncm.json` | `nenhum`, 0 épocas | **Referência sem treino**: cosseno entre `phi` e a média de `phi` de cada classe (SimpleCIL). Todas as classes no mesmo espaço, sem vazamento entre Adapters por construção. |
| `config_baseline.json` | `nenhum` | Linha de base: Adapters treinados, sem pseudo-negativos |
| `config_gauss.json` | `gauss` | Pseudo-negativos sorteados às cegas do banco (`beta = 0`): o replay gaussiano já conhecido (PASS, SLCA, DMR) |
| `config_experimento.json` | `is` | Pseudo-negativos escolhidos pela **margem de vazamento** (amostragem por importância): a proposta |

Contrastes: `baseline → gauss` mede o que já é conhecido; `gauss → experimento` mede o ganho da
escolha guiada pela margem; `ncm` diz se os Adapters treinados ajudam ou atrapalham o ponto de partida.

Os três braços treinados são idênticos exceto pelo modo. CIFAR-100, 5 tarefas × 20 classes,
dataset em `../../my_data`, seeds 1993 a 1996.

### Amostradores (`pseudo_negativos.modo`)

- `gauss`: sorteia a classe antiga e um `phi` da gaussiana dela.
- `is`: sorteia `n_proposta` (4096) candidatos do banco, calcula a margem
  `melhor score de classe nova − score da própria classe antiga` e reamostra com peso
  `exp(beta · margem)`. Concentra os exemplos onde o Adapter novo está vazando.
- `gfn`: GFlowNet híbrida (classe discreta, depois `phi` contínuo), treinada junto com o Adapter.
  Implementada, mas fora do estudo: só se justifica se o ESS do `is` cair perto de zero.

## Opções de treino (`treino.*` no config)

Sem a chave no config, vale o comportamento antigo (o das primeiras execuções; ver o histórico de
commits). Os quatro configs usam os valores da coluna "estudo".

| Opção | Antigo | Estudo | O que faz |
| --- | --- | --- | --- |
| `adapter_residual` | `false` | `true` | Adapter devolve `phi + up(relu(down(phi)))` e começa como identidade (com `false`, devolve só `up(...)`, começa em zero) |
| `escala_logits` | `1.0` | `16.0` | Multiplica os cossenos antes da entropia cruzada (reais e pseudo-negativos). Com 1, a softmax nunca fica confiante (NormFace). Não muda a predição |
| `init_classes_novas` | `"aleatorio"` | `"media_phi"` | Classes novas começam na média das saídas do Adapter (= média de `phi`, com Adapter residual), em vez de vetores sorteados |
| `atualizar_prototipos_novos` | `false` | `true` | A cada época, as classes novas passam a ser a média das saídas do Adapter: treino e inferência usam o mesmo alvo |
| `alpha_lvq`, `lambda_lvq` | `0.4`, `0.3` | `0`, `0` | LVQ desligado (as unidades começavam na origem e nunca se moviam) |
| `parada_antecipada.ativo` | `false` | `true` | Ver abaixo |

### Parada antecipada

O **teste nunca decide nada**. De cada classe do treino, 10% são separados como validação
(estratificado, fixo pela seed); essas imagens não entram no treino, nos protótipos nem no banco.

| Chave | Valor | Significado |
| --- | --- | --- |
| `fracao_validacao` | `0.1` | fração do treino de cada classe usada como validação |
| `metrica` | `"ambas"` | acompanha **separadamente** (1) a CE de validação nas classes novas (aprender o novo) e (2) a CE de um conjunto fixo de pseudo-negativos sorteados às cegas (recusar o antigo); só para quando **nenhuma das duas** melhora há `paciencia` épocas. Igual em todos os braços, inclusive a linha de base (na tarefa 0, só a parte 1). Alternativas: `"val_ce"` (só o novo) e `"val_ce_mais_pn"` (soma com recorde único) |
| `peso_pn_validacao` | `0.5` | peso da parte 2 na soma usada para escolher a época restaurada (independe de `lambda_pn`, que é peso de treino) |
| `paciencia` | `10` | épocas seguidas sem melhora antes de parar |
| `delta_minimo` | `0.001` | melhora mínima |
| `epocas_minimas` | `5` | nunca para antes disto |
| `restaurar_melhor` | `true` | ao parar, volta ao Adapter e aos protótipos da melhor época da **janela final** (últimas `paciencia + 1` épocas), pela soma das duas partes |
| `pseudo_negativos_validacao` | `1000` | conjunto fixo de pseudo-negativos de validação (ver H4) |

`epochs_por_tarefa` (100) passa a ser o máximo.

**Por que `"ambas"`.** O Adapter novo aprende duas coisas, em tempos diferentes: primeiro a
reconhecer as classes novas; a recusa das antigas **primeiro piora** (ele "agarra" tudo) e só depois
melhora. Na seed 1993, a regra `val_ce` (só o novo) parava e restaurava épocas em que a recusa ainda
não tinha sido aprendida (braço gauss, tarefa 1: 57% dos pseudo-negativos fixos vazavam na época
restaurada e 25% cinco épocas depois), e a acurácia final caiu de 33,2% (100 épocas) para 28,2%. Uma
soma das duas partes com um recorde único também falha: o mínimo fica nas primeiras épocas, antes de
o Adapter agarrar, e a simulação nos logs da seed 1993 mostrou que ela pararia ainda mais cedo
(época 11, restaurando a época 1). Com `"ambas"`, cada parte tem seu recorde e o treino só para
quando as duas estabilizam; a restauração olha só a janela final, para não voltar a essas épocas.

## Como rodar

```bash
bash rodar_experimentos.sh --teste       # primeiro: seed 1993, poucas épocas, força parar e restaurar
bash rodar_experimentos.sh --bateria 1   # seed 1993: ncm, linha de base, experimento, gauss
bash rodar_experimentos.sh --bateria 2   # seed 1994 (3: 1995, 4: 1996)
bash rodar_experimentos.sh --bateria 1 2 # várias em sequência
```

- Execuções concluídas deixam um marcador `resultados/execucoes/<config>_seed<seed>.ok` e são
  puladas se o script for chamado de novo (`--refazer` força rodar tudo).
- No fim de cada bateria, `comparar_resultados.py` compara todas as seeds já concluídas.
- Windows: Git Bash ou WSL.

Execução isolada: `python cil_nova_estrategia_2.py --config config_experimento.json --seed 1993`
(sem `--config`, usa `config_experimento.json`).

## Onde ficam os resultados

```
resultados/
├── execucoes/     saída do terminal de cada execução, marcadores .ok, resumo.txt
├── logs/          log detalhado de cada execução
├── results/       JSON com todos os números de cada execução
├── figures_cil/   uma pasta de gráficos por execução (fora do Git)
├── checkpoints/   modelos treinados (fora do Git)
└── comparacao/    gráficos e resumo.csv comparando os braços
```

Cada execução é identificada por `<config>_seed<seed>_<data_hora>`; nada é sobrescrito. A
comparação usa a execução mais recente de cada (config, seed).

### Gráficos comparativos (`resultados/comparacao/<data_hora>/`)

| Arquivo | O que mostra |
| --- | --- |
| `1_pares_por_seed.png` | acurácia média final, vazamento, erro dentro da tarefa e teto de ganho, por seed |
| `2_vazamento_por_fronteira.png` | vazamento após cada tarefa (média e faixa entre seeds) |
| `3_acc_final_por_tarefa.png` | acurácia final de cada tarefa: onde o esquecimento se concentra |
| `4_diagnostico_pseudo_negativos.png` | experimento: fração dos pseudo-negativos escolhidos que vazam e ESS por época |
| `5_h4_pseudo_negativos_validacao.png` | H4 nos três braços: fração de um conjunto **fixo** de pseudo-negativos sorteados às cegas que vazaria, por época |
| `resumo.csv` | os números, inclusive AAC incremental e a linha de referência NCM |

## O que olhar

- **Acurácia média final** (`aac` no JSON, nome mantido por compatibilidade) e **AAC incremental**
  (`aac_incremental`: média das acurácias no teste acumulado após cada tarefa, a AAC da literatura).
- **Vazamento, sempre comparado ao NCM.** Também o índice de direção (> 1 = vazamento para tarefas
  mais novas) e a parcela atribuível à escala (quanto some só padronizando os scores de cada Adapter).
- **Acurácia com oráculo de tarefa**: o quanto os Adapters ajudam dentro da tarefa.
- **Diagnóstico de aritmética de tarefas** (por fronteira): acurácia nas tarefas antigas com o
  Adapter novo (original), sem a tarefa nova (exclusão), com o Adapter novo negado e com um Adapter
  aleatório. `exclusão − original` = quanto a tarefa nova "rouba" das antigas.
- **Logs por época** (a cada `frequencia_log_epocas` e na parada): `[Validação]` com a CE de
  validação, a melhor época até ali e a fração de pseudo-negativos fixos que vazam; `[Pseudo-negativos]`
  com ESS/propostas (perto de 0 = o `is` deixa de bastar).

### Hipóteses

- **H1** gauss reduz o vazamento em relação à linha de base.
- **H2** experimento reduz o vazamento em relação ao gauss.
- **H3** os ganhos concentram-se no vazamento (o erro dentro da tarefa muda pouco).
- **H4** a fração de pseudo-negativos fixos que vazam cai durante o treino de cada tarefa.

## Limitações conhecidas

- **Cenário**: ResNet-18 com imagens 64×64 e um único conjunto de dados, abaixo do padrão da
  literatura com modelos pré-treinados (ViT-B/16 a 224×224, vários conjuntos).
- **Modelo gaussiano**: covariância de posto 32 e ReLU aplicada às amostras; classes multimodais
  seriam mais bem descritas por misturas.
- **Validação**: com a parada antecipada, cada classe treina com 90% das imagens.
- **Estatística**: 4 seeds dão comparações pareadas com pouco poder (Wilcoxon bilateral: p mínimo 0,125).
- **Modo `gfn` em 512 dimensões**: não avaliado no CIFAR-100.

## Arquivos

| Arquivo | Conteúdo |
| --- | --- |
| `cil_nova_estrategia_2.py` | treino, avaliação, diagnósticos e gráficos de cada execução |
| `pseudo_negativos_cil.py` | banco de gaussianas, margem, amostradores `gauss`, `is`, `gfn` |
| `metricas_vazamento.py` | métricas de vazamento |
| `comparar_resultados.py` | comparação pareada entre braços e `resumo.csv` |
| `rodar_experimentos.sh` | baterias por seed |
| `config_ncm.json`, `config_baseline.json`, `config_gauss.json`, `config_experimento.json` | os quatro configs do estudo |
| `cil_nova_estrategia_1.py`, `config.json` | Estratégia 1 (referência anterior), inalterada |
