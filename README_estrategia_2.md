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

## Os configs do estudo

| Config | `pseudo_negativos.modo` | Papel |
| --- | --- | --- |
| `config_ncm.json` | `nenhum`, 0 épocas | **Referência sem treino**: cosseno entre `phi` e a média de `phi` de cada classe (SimpleCIL). Todas as classes no mesmo espaço, sem vazamento entre Adapters por construção. |
| `config_baseline.json` | `nenhum` | Linha de base: Adapters treinados, sem pseudo-negativos |
| `config_gauss.json` | `gauss` | Pseudo-negativos sorteados às cegas do banco (`beta = 0`): o replay gaussiano já conhecido (PASS, SLCA, DMR) |
| `config_experimento.json` | `is` | Pseudo-negativos escolhidos pela **margem de vazamento** (amostragem por importância), margem nas notas originais |
| `config_margem_padronizada.json` | `is`, `margem_padronizada = true`, `beta = 0.5` | Igual ao experimento, mas a margem usa as **notas padronizadas**, as mesmas da decisão final |

Contrastes: `baseline → gauss` mede o que já é conhecido; `gauss → experimento` e
`gauss → margem_padronizada` medem o ganho da escolha guiada pela margem (H2), nas notas originais e
nas padronizadas; `ncm` diz se os Adapters treinados ajudam ou atrapalham o ponto de partida.

### Notas padronizadas

Cada Adapter dá notas numa escala própria, e os mais novos tendem a dar notas mais altas para tudo,
o que puxa as imagens para as tarefas novas. **Padronizar** é ler cada nota como "quanto ela está
acima do que esse Adapter costuma dar": `(nota − média_k) / desvio_k`, com média e desvio da nota
vencedora nas imagens de **treino** da tarefa k (o teste não entra). Não muda o tipo vencedor dentro
de cada tarefa, só qual tarefa vence. Nas 4 seeds do CIFAR-100 (avaliadas pelos checkpoints, com
`avaliar_padronizacao.py`), levou gauss e experimento de cerca de 30,6% para 43,5% de acurácia
final, acima dos 41,2% do NCM.

O JSON de cada execução traz as duas versões: `resultado_final`, `metricas_vazamento_final`,
`metricas_vazamento_por_fronteira` e `acc_apos_tarefa` (notas originais) e os mesmos campos com
`_padronizado`/`_padronizada` (notas padronizadas). `versao_principal` diz qual é a principal.

Os três braços treinados são idênticos exceto pelo modo. Por padrão, CIFAR-100, 5 tarefas × 20
classes, dataset em `../../my_data`, seeds 1993 a 1998 (baterias 1 a 6); outros datasets por perfil (abaixo).

### Amostradores (`pseudo_negativos.modo`)

- `gauss`: sorteia a classe antiga e um `phi` da gaussiana dela.
- `is`: sorteia `n_proposta` (4096) candidatos do banco, calcula a margem
  `melhor score de classe nova − score da própria classe antiga` e reamostra com peso
  `exp(beta · margem)`. Concentra os exemplos onde o Adapter novo está vazando.
- `gfn`: GFlowNet híbrida (classe discreta, depois `phi` contínuo), treinada junto com o Adapter.
  Implementada, mas fora do estudo: só se justifica se o ESS do `is` cair perto de zero.

## Opções de treino (`treino.*` no config)

Sem a chave no config, vale o comportamento antigo (o das primeiras execuções; ver o histórico de
commits). Os configs do estudo usam os valores da coluna "estudo".

| Opção | Antigo | Estudo | O que faz |
| --- | --- | --- | --- |
| `adapter_residual` | `false` | `true` | Adapter devolve `phi + up(relu(down(phi)))` e começa como identidade (com `false`, devolve só `up(...)`, começa em zero) |
| `escala_logits` | `1.0` | `16.0` | Multiplica os cossenos antes da entropia cruzada (reais e pseudo-negativos). Com 1, a softmax nunca fica confiante (NormFace). Não muda a predição |
| `init_classes_novas` | `"aleatorio"` | `"media_phi"` | Classes novas começam na média das saídas do Adapter (= média de `phi`, com Adapter residual), em vez de vetores sorteados |
| `atualizar_prototipos_novos` | `false` | `true` | A cada época, as classes novas passam a ser a média das saídas do Adapter: treino e inferência usam o mesmo alvo |
| `alpha_lvq`, `lambda_lvq` | `0.4`, `0.3` | `0`, `0` | LVQ desligado (as unidades começavam na origem e nunca se moviam) |
| `padronizar_notas` | `false` | `true` | A versão padronizada vira a principal (as duas são sempre calculadas e gravadas) |
| `cache_phi` | `false` | `true` | Calcula `phi = backbone(imagem)` **uma vez** e treina os Adapters sobre ele. O backbone é congelado e não há aumento de dados, então é a mesma conta, muito mais rápida |
| `cache_phi_em_disco` | — | `true` | Guarda `phi` em `resultados/cache_phi/`, reaproveitado pelas outras execuções do mesmo dataset (fora do Git) |
| (sempre, com `cache_phi`) | — | — | `phi` fica inteiro na GPU e os lotes são fatias dele (sem custo de Python por amostra); as notas de cosseno são calculadas com uma multiplicação de matrizes por tarefa; o banco de gaussianas sorteia todas as classes de uma vez; com a LVQ desligada, ela não é calculada. Mesmas contas; os números mudam só na ordem dos sorteios |
| `avaliar_teste_a_cada` | `1` | `5` | Teste durante o treino (só para os gráficos) a cada N épocas, e sempre na 1ª, na última e nas de log |
| `parada_antecipada.ativo` | `false` | `false` | Ver abaixo. Desligada no estudo: com `phi` pré-calculado, 100 épocas fixas custam pouco e cada braço treina o mesmo tempo |

Em `pseudo_negativos.*`: `margem_padronizada` (padrão `false`) faz a margem dos pseudo-negativos usar
as notas padronizadas. As estatísticas das tarefas antigas são as definitivas; as da tarefa atual são
uma média móvel da nota vencedora nas imagens de treino. A margem passa a ficar em unidades de desvio
(cerca de 10 vezes maior que em cosseno), por isso esse config usa `beta = 0.5`.

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

## Datasets e tamanho das tarefas

Os configs usam o CIFAR-100. Para outro dataset, **não edite os configs**: use um **perfil**
(`perfis/<nome>.json`), que substitui só os blocos `dataset`, `tarefas` e `avaliacao_extra`. Os
braços continuam iguais, e as saídas levam o nome do perfil (ex.: `gauss_imagenet_r_seed1993_...`),
então o `comparar_resultados.py` compara cada dataset no seu próprio grupo.

```bash
ls perfis/                                              # perfis disponíveis
bash rodar_experimentos.sh --perfil imagenet_r --teste
bash rodar_experimentos.sh --perfil imagenet_r --bateria 1 2 3 4
python avaliar_padronizacao.py --perfil imagenet_r      # padronização das notas desse grupo
```

| Perfil | Classes | Tarefas (padrão) | Entrada | Onde o código procura |
| --- | --- | --- | --- | --- |
| `cifar100` | 100 | 5 × 20 | 64×64 | `../../my_data/cifar-100-python/` |
| `imagenet_r` | 200 | 10 × 20 | 224×224, recorte central | `../../my_data/imagenet-r/{train,test}/` |
| `imagenet_a` | 200 | 10 × 20 | 224×224, recorte central | `../../my_data/imagenet-a/{train,test}/` |
| `objectnet` | 200 | 10 × 20 | 224×224, recorte central | `../../my_data/objectnet/{train,test}/` |
| `omnibenchmark` | 300 | 10 × 30 | 224×224, recorte central | `../../my_data/omnibenchmark/{train,test}/` |
| `vtab` | 50 | 5 × 10 | 224×224, recorte central | `../../my_data/vtab/{train,test}/` |

O número de classes é o das versões distribuídas com SimpleCIL/APER e EASE (Zhou et al.); se as
pastas tiverem outro número, o log avisa. Os tamanhos de tarefa são um ponto de partida: os artigos
da área usam vários (ex.: ImageNet-R com 5, 10 ou 20 classes por tarefa) e convém conferir o
protocolo de cada artigo com que se quer comparar.

### Bloco `dataset`

| Chave | Significado |
| --- | --- |
| `nome` | `cifar10`, `cifar100`, `imagenet_r`, `imagenet_a`, `objectnet`, `omnibenchmark`, `vtab` ou `pastas` (qualquer dataset com uma subpasta por classe) |
| `raiz` | CIFAR: a pasta que **contém** `cifar-100-python/`. Demais: a pasta que contém as subpastas de treino e teste. Relativa à pasta do projeto |
| `pasta_treino`, `pasta_teste` | subpastas de treino e teste (padrão `train` e `test`). O teste é lido com os índices de classe do treino |
| `pasta_unica`, `fracao_teste`, `seed_divisao` | para dataset **sem** divisão em treino e teste: separa `fracao_teste` de cada classe para teste, de forma fixa. Resultados só comparáveis com a literatura se a divisão for a mesma |
| `resize` | `[altura, largura]` da entrada |
| `recorte_central` | `true` para imagens não quadradas (redimensiona o lado menor e recorta o centro); `false` redimensiona direto |

### Bloco `tarefas`

| Chave | Significado |
| --- | --- |
| `classes_por_tarefa` | tamanho de cada tarefa incremental |
| `num_tasks` | número de tarefas, ou `"auto"` = todas as classes do dataset divididas em tarefas de `classes_por_tarefa` (protocolo "B0 IncX"); classes que sobram ficam de fora |
| `ordem_classes` | `"aleatoria"` (sorteada pela seed, padrão) ou `"original"` (ordem dos índices do dataset) |

Todas as tarefas têm o mesmo tamanho (não há tarefa inicial maior, como em "B50 Inc10").

### Só testar em outro dataset (`avaliacao_extra`)

Para avaliar o modelo final num dataset em que ele **não treinou** (ex.: ObjectNet), acrescente ao
perfil:

```json
"avaliacao_extra": [
    {"nome": "objectnet", "raiz": "../../my_data/objectnet", "pasta": "test",
     "mapa_classes": "perfis/mapa_objectnet_para_imagenet_r.json"}
]
```

Só entram as imagens das classes que existem no treino, e o modelo escolhe entre todas as classes
treinadas. Sem `mapa_classes`, casa nomes de pasta iguais (sem diferenciar maiúsculas, espaços e
hífens). Quando os nomes diferem (ex.: pastas do ImageNet-R com códigos `n0...` e do ObjectNet com
nomes de objetos), o mapa é um JSON `{"pasta_no_extra": "pasta_no_treino", ...}`. O resultado vai
para `avaliacao_extra` no JSON da execução, com quantas classes casaram, acurácia e vazamento.

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
├── cache_phi/     phi pré-calculado por dataset (fora do Git)
├── padronizacao/  saída de avaliar_padronizacao.py
└── comparacao/    gráficos e resumo.csv comparando os braços
```

Cada execução é identificada por `<config>_seed<seed>_<data_hora>`; nada é sobrescrito. A
comparação usa a execução mais recente de cada (config, seed).

### Gráficos comparativos (`resultados/comparacao/<data_hora>/` e `<data_hora>_padronizada/`)

A comparação é gerada duas vezes: com as notas originais e, na pasta com sufixo `_padronizada`, com
as notas padronizadas. A referência NCM aparece nas duas com as notas originais.

| Arquivo | O que mostra |
| --- | --- |
| `1_pares_por_seed.png` | acurácia média final, vazamento, erro dentro da tarefa e teto de ganho, por seed |
| `2_vazamento_por_fronteira.png` | vazamento após cada tarefa (média e faixa entre seeds) |
| `3_acc_final_por_tarefa.png` | acurácia final de cada tarefa: onde o esquecimento se concentra |
| `4_diagnostico_pseudo_negativos.png` | experimento: fração dos pseudo-negativos escolhidos que vazam e ESS por época |
| `5_h4_pseudo_negativos_validacao.png` | H4: fração de um conjunto **fixo** de pseudo-negativos sorteados às cegas que vazaria, por época (só com parada antecipada ativa) |
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
- **Estatística**: com 6 seeds, o teste de Wilcoxon pareado bilateral consegue chegar a p = 0,031 (com 4, o mínimo é 0,125).
- **Modo `gfn` em 512 dimensões**: não avaliado no CIFAR-100.

## Arquivos

| Arquivo | Conteúdo |
| --- | --- |
| `cil_nova_estrategia_2.py` | treino, avaliação, diagnósticos e gráficos de cada execução |
| `pseudo_negativos_cil.py` | banco de gaussianas, margem, amostradores `gauss`, `is`, `gfn` |
| `metricas_vazamento.py` | métricas de vazamento |
| `comparar_resultados.py` | comparação pareada entre braços e `resumo.csv` |
| `avaliar_padronizacao.py` | avalia, a partir dos checkpoints e sem treinar, o efeito de padronizar as notas de cada Adapter (média e desvio do treino da tarefa) na acurácia e no vazamento |
| `rodar_experimentos.sh` | baterias por seed |
| `config_ncm.json`, `config_baseline.json`, `config_gauss.json`, `config_experimento.json`, `config_margem_padronizada.json` | os configs do estudo |
| `perfis/*.json` | um perfil por dataset (bloco `dataset`, tamanho das tarefas, avaliação extra) |
| `cil_nova_estrategia_1.py`, `config.json` | Estratégia 1 (referência anterior), inalterada |
