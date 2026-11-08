# Nova Estratégia 2: pseudo-negativos no espaço do backbone congelado

A Estratégia 2 é a Nova Estratégia 1 com uma peça a mais: durante o treino de cada tarefa nova,
o Adapter novo passa a ver **exemplos sintéticos das classes antigas** e aprende a recusá-los.
A Estratégia 1 continua intacta e serve de linha de base.

## O problema que isto ataca

No treino da tarefa t (`treinar_tarefa`), a perda CE só vê imagens das classes novas. Os Adapters
antigos estão congelados. Logo, **nada ensina o Adapter_t a dar score baixo para uma imagem de
classe antiga**: ele nunca vê uma. Na inferência, uma imagem antiga recebe score do Adapter antigo
(para a classe certa) e do Adapter_t (para as classes novas), e às vezes o segundo ganha. É o vazamento.

## Por que a solução é simples nesta arquitetura

O Adapter de cada tarefa recebe `phi = backbone(imagem)`, a saída da ResNet-18 **congelada**.
Portanto:

- o `phi` de uma classe antiga nunca muda (o drift semântico ~ 0 dos logs confirma);
- basta guardar, ao fim de cada tarefa, a **média e a covariância de `phi`** de cada classe;
- um `phi` sorteado dessa gaussiana pode ser passado pelos Adapters exatamente como um real.

Nenhuma imagem é guardada: continua sem replay de exemplos. É a mesma família de ideias do replay
gaussiano de features (PASS, Zhu et al., CVPR 2021) e do alinhamento de classificador do SLCA
(Zhang et al., ICCV 2023).

## O que mudou no treino

```
perda = (1 - lambda_lvq) * CE_real + lambda_lvq * L_LVQ + lambda_pn * CE_pseudo
```

`CE_pseudo` usa os `phi` sintéticos com o **rótulo antigo**, avaliados como na inferência
(cosseno + LVQ). Como os Adapters e as unidades LVQ antigas não recebem gradiente, o único jeito
de reduzir `CE_pseudo` é **baixar o score das classes novas** para esses `phi`.

## Como os `phi` sintéticos são escolhidos (`pseudo_negativos.modo`)

| Modo | O que faz | Do brinquedo 2D |
| --- | --- | --- |
| `nenhum` | sem pseudo-negativos: reproduz a Estratégia 1 | — |
| `gauss` | sorteio direto das gaussianas antigas | — |
| `is` | sorteia muitas propostas e reamostra com peso `exp(beta * margem)`, em que `margem = melhor score de classe nova - score da própria classe antiga`; concentra os exemplos **onde hoje há vazamento** | amostragem por importância, o melhor resultado |
| `gfn` | GFlowNet híbrida: escolhe a classe antiga, depois gera `phi` em poucos passos; treinada junto com o Adapter, com buffer do `is` | V3 |

Correspondência com o brinquedo (`pseudo_negativos_gfn.py`):

| No brinquedo 2D | Neste projeto |
| --- | --- |
| ponto no plano | `phi` (512 dimensões) |
| manchas A, B, C | gaussianas das classes de tarefas anteriores (`BancoGaussianas`) |
| estrelas | classes da tarefa atual (`fc[c]` e unidades LVQ, vistos pelo Adapter_t) |
| `score_t(z)` | `margem_vazamento`: quanto o Adapter_t "rouba" o ponto da classe antiga |
| `r(z) = p_antiga(z) * exp(beta * score)` | igual, com a margem no lugar do score |
| uso dos pontos | CE com rótulo antigo no treino do Adapter_t |

## Os três braços

| Config | `pseudo_negativos.modo` | Papel |
| --- | --- | --- |
| `config_baseline.json` | `nenhum` | linha de base, sem pseudo-negativos |
| `config_gauss.json` | `gauss` | pseudo-negativos sorteados às cegas: o replay gaussiano já conhecido (PASS, SLCA, DMR) |
| `config_experimento.json` | `is` | pseudo-negativos escolhidos pela margem de vazamento: a proposta |

- `baseline -> gauss` mede o ganho do que já é conhecido;
- `gauss -> experimento` mede o ganho da escolha guiada pela margem (`gauss` é o caso `beta = 0`).

Cada bateria roda os três braços da sua seed. Execuções já concluídas são puladas: uma bateria
rodada antes da existência do `config_gauss.json` executa só o braço `gauss` quando chamada de novo.
`comparar_resultados.py` compara os três braços nas seeds em que os três existem e, se houver seeds
sem o `gauss`, também linha de base x experimento em todas as seeds (pasta com sufixo `_2bracos`).

## Correções v2

A análise da bateria 1 (seed 1993, linha de base e experimento) mostrou que 90,6% do erro final é
vazamento, mas também que o treino quase não aprende e que os pseudo-negativos mal agem. As causas
estão no código herdado da Estratégia 1 e foram corrigidas como **opções de config**. Os valores
padrão reproduzem o comportamento anterior, então os configs antigos continuam dando os mesmos
resultados.

| Opção (`treino.`) | Padrão (antigo) | v2 | O que corrige |
| --- | --- | --- | --- |
| `escala_logits` | `1.0` | `16.0` | Cossenos em [-1, 1] sem escala: a softmax nunca fica confiante e a entropia cruzada fica colada em ln(N) (NormFace). Vale para as imagens reais e para os pseudo-negativos. Não muda a predição. |
| `init_classes_novas` | `"aleatorio"` | `"media_phi"` | Durante o treino, as classes novas eram vetores sorteados, trocados pela média da classe só no fim. A margem dos pseudo-negativos era medida contra esses vetores. |
| `atualizar_prototipos_novos` | `false` | `true` | Ao fim de cada época, as classes novas passam a ser a média das saídas do Adapter nas imagens da época. |
| `alpha_lvq`, `lambda_lvq` | `0.4`, `0.3` | `0`, `0` | As unidades LVQ começam perto da origem e nunca saem de lá: a perda LVQ fica em ~1e-11 e o score LVQ em ~0. Desligado. |
| `adapter_residual` | `false` | `true` | O Adapter devolvia só `up(h)` (começa em zero, cada tarefa num espaço próprio de até 64 dimensões). Com `true`, devolve `phi + up(h)` e começa como identidade. |

Também mudou, sem opção (são correções de diagnóstico, não do treino):

- **Diagnóstico de aritmética de tarefas**: as versões original, negação e aleatório recebiam só
  as classes antigas, então o Adapter novo nunca era usado e as quatro versões davam o mesmo
  número. Agora as classes novas competem nessas três versões.
- **AAC incremental**: o campo `aac` do JSON é a acurácia média final entre tarefas. Agora o JSON
  também traz `acc_apos_tarefa` (acurácia no teste acumulado após cada tarefa, inclusive a 0) e
  `resultado_final.aac_incremental` (a média delas, a AAC da literatura). O
  `comparar_resultados.py` mostra as duas e reconstrói a incremental para execuções antigas.
- **`epochs_por_tarefa = 0`** agora funciona (uma avaliação sem treino).

### Configs v2

| Config | Papel |
| --- | --- |
| `config_ncm_v2.json` | Referência sem treino: cosseno entre phi e a média de phi de cada classe (SimpleCIL). Não tem vazamento entre Adapters por construção. Leva minutos. |
| `config_baseline_v2.json` | Linha de base v2, sem pseudo-negativos |
| `config_gauss_v2.json` | Pseudo-negativos às cegas, v2 |
| `config_experimento_v2.json` | Pseudo-negativos pela margem, v2 |

```bash
bash rodar_experimentos.sh --v2 --teste       # primeiro: seed 1993, 2 épocas (ncm sem treino)
bash rodar_experimentos.sh --v2 --bateria 1   # seed 1993, os quatro configs v2
```

Os resultados v2 ficam em `resultados/`, com `_v2` no nome, e o `comparar_resultados.py` os compara
num grupo separado (pasta de comparação com sufixo `_v2`), com a referência NCM no `resumo.csv`.

**O que olhar primeiro:** se `ncm_v2` tiver acurácia maior que `baseline_v2`, os Adapters treinados
estão piorando o ponto de partida, e isso precisa ser discutido antes dos pseudo-negativos.

## Como rodar tudo de uma vez

```bash
bash rodar_experimentos.sh --teste       # primeiro: seed 1993, 2 épocas, linha de base + experimento
bash rodar_experimentos.sh --bateria 1   # seed 1993: linha de base + experimento, 100 épocas
bash rodar_experimentos.sh --bateria 2   # seed 1994
bash rodar_experimentos.sh --bateria 3   # seed 1995
bash rodar_experimentos.sh --bateria 4   # seed 1996
bash rodar_experimentos.sh --bateria 1 2 # várias baterias em sequência, se quiser
```

Uma bateria por seed. A comparação no fim de cada bateria usa todas as seeds já concluídas:
depois da bateria 1, só a seed 1993; depois da 2, 1993 e 1994; e assim por diante.

- `config_baseline.json`: `pseudo_negativos.modo = "nenhum"` (linha de base).
- `config_experimento.json`: `pseudo_negativos.modo = "is"` (experimento).
- Os dois são idênticos no resto: CIFAR-100, 5 tarefas x 20 classes, dataset em `../../my_data`.
- A saída de cada execução vai para `resultados/execucoes/<config>_seed<seed>.txt`, e um resumo
  acumulado para `resultados/execucoes/resumo.txt`.
- Se o script for interrompido, basta chamá-lo de novo: o que terminou é pulado
  (`--refazer` força rodar tudo de novo).
- Windows: rodar no Git Bash ou no WSL.

## Onde ficam os resultados

Tudo dentro do projeto, em `CIL_e_GFlowNet/resultados/`, qualquer que seja a pasta de onde o
comando é executado:

```
CIL_e_GFlowNet/resultados/
├── execucoes/     saída do terminal de cada execução + resumo.txt
├── logs/          log detalhado de cada execução
├── results/       JSON com todos os números de cada execução
├── figures_cil/   uma pasta de gráficos por execução
├── checkpoints/   modelos treinados
└── comparacao/    gráficos linha de base x experimento
```

## Gráficos

**Por execução** (os mesmos da Estratégia 1, mais os de 100 classes em blocos por tarefa), numa
pasta própria: `resultados/figures_cil/<config>_seed<seed>_<data_hora>/`. Log, JSON e checkpoint levam o
mesmo identificador no nome, então nada é sobrescrito.

**Comparativos**, gerados por `comparar_resultados.py` (chamado automaticamente no fim do
`rodar_experimentos.sh`, ou à mão com `python comparar_resultados.py`), em
`resultados/comparacao/<data_hora>/`:

| Arquivo | O que mostra |
| --- | --- |
| `1_pares_por_seed.png` | AAC, vazamento, erro dentro da tarefa e teto de ganho; cada seed é uma linha ligando linha de base e experimento, com a diferença média e em quantas seeds melhorou |
| `2_vazamento_por_fronteira.png` | vazamento após cada tarefa, média e faixa (mínimo a máximo) entre seeds |
| `3_acc_final_por_tarefa.png` | acurácia final de cada tarefa: onde o esquecimento se concentra |
| `4_diagnostico_pseudo_negativos.png` | só experimento: fração de pseudo-negativos que vazam (deve cair) e ESS por época |
| `resumo.csv` | os números de todos os gráficos, separados por `;` (abre no Excel) |

Usa a execução mais recente de cada (config, seed) e só as seeds com as duas execuções concluídas.

## Como rodar uma execução isolada

```bash
# linha de base (equivale à Estratégia 1)
python cil_nova_estrategia_2.py   # com "modo": "nenhum" em config_estrategia_2.json
# pseudo-negativos por amostragem por importância (padrão do config)
python cil_nova_estrategia_2.py --seed 1993
```

**Configuração padrão: CIFAR-100, 5 tarefas de 20 classes** (todas as 100 classes). Como todas
as classes entram, a seed muda só a ordem e o agrupamento das classes em tarefas.

Primeiro teste de execução: `epochs_por_tarefa = 2` no config, mantendo 5 x 20. Isso já passa por
todas as fronteiras, pelo banco com 80 classes antigas e pelos gráficos de 100 classes.

Custo: a avaliação no teste acumulado roda a cada época (como na Estratégia 1), então com 100
épocas por tarefa são 500 avaliações sobre até 10 mil imagens. Pesa mais do que nas 3 x 2 originais.

**Este código não foi executado no ambiente em que foi escrito (PyTorch indisponível lá).**
A sintaxe foi verificada, e as fórmulas da gaussiana de posto baixo foram conferidas em NumPy
contra uma gaussiana de covariância cheia. O restante precisa do teste rápido acima.

## O que olhar nos logs

A cada `frequencia_log_epocas`, para as tarefas t >= 1:

```
[Pseudo-negativos] CE_pn | vazam (margem>0) | margem média | ESS/propostas | logZ IS | logZ GFN | perda TB
```

- **vazam (margem > 0)**: fração dos pseudo-negativos que vazariam. Deve **cair** ao longo das
  épocas: é o Adapter_t aprendendo a recusá-los.
- **ESS/propostas** (modos `is` e `gfn`): o diagnóstico que decide se a GFlowNet vale a pena.
  Uma fração razoável significa que a amostragem por importância basta. Perto de 0 significa que
  as regiões de vazamento estão na cauda das gaussianas, e aí faz sentido testar `gfn`.
- **logZ GFN abaixo de logZ IS** (modo `gfn`): a GFlowNet está deixando massa para trás.

Após cada fronteira e no final, `MÉTRICAS DE VAZAMENTO` (de `metricas_vazamento.py`), também
gravadas no JSON em `metricas_vazamento_por_fronteira` e `metricas_vazamento_final`:

- `vazamento` e a decomposição exata `1 - acc = vazamento + erro_dentro_da_tarefa`;
- `teto_de_ganho`: quanto se ganharia zerando o vazamento;
- `matriz_L` e `indice_direcao`: para onde o vazamento vai (> 1 = para tarefas mais novas);
- `parcela_atribuivel_a_escala`: quanto do vazamento some só normalizando os scores por tarefa.

A comparação principal é `nenhum` contra `is`, nas 4 seeds: o vazamento e o AAC final.

## Limitações conhecidas

- **ReLU**: `phi` da ResNet-18 é não negativo; a gaussiana não. Com `relu = true` (padrão), os
  `phi` sintéticos passam por ReLU antes do uso, o que deforma um pouco a distribuição.
- **Covariância de posto baixo**: com `rank_cov = 32`, o banco captura só parte da variância
  (o resto vira um termo isotrópico). Aumentar o posto custa memória e pede mais amostras por classe.
- **Só CE para os pseudo-negativos**: a perda LVQ não é aplicada a eles, porque as unidades LVQ
  antigas vivem no espaço dos Adapters antigos, não no do Adapter_t.
- **Escala do problema**: com 5 tarefas, a matriz L tem 20 entradas fora da diagonal. As
  correlações de recência e semântica já dizem algo, mas ainda com pouca precisão: compare-as
  entre as 4 seeds antes de concluir.
- **Gráficos com mais de 20 classes**: matriz de confusão e mapa de calor aparecem em blocos por
  tarefa (blocos fora da diagonal = vazamento entre tarefas), o t-SNE é colorido por tarefa, e as
  barras de margem e de drift perdem os rótulos de valor. Com até 20 classes, tudo fica como na
  Estratégia 1.
- **Modo `gfn` em 512 dimensões**: os testes de Lahlou et al. (2023) com recompensa vão até 10
  dimensões. Use só depois de o ESS indicar que o `is` não basta.
