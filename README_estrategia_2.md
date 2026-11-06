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

## Como rodar

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
