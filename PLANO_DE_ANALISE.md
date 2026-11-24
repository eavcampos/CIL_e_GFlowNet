# Plano de análise — Vazamento entre adapters em aprendizado incremental de classes sem exemplares

**Versão:** 1.0, 24/11/2026, revisada e aprovada por Edilene e Takashi. A partir desta versão, o texto não se altera; qualquer mudança entra como adendo datado (Seção 10).
**Autores:** Edilene Aparecida Veneruchi de Campos, Edson Takashi Matsubara (FACOM-UFMS).
**Onde fica:** versionado no repositório Git do projeto, junto com o código (Seção 9). Não foi feito registro em plataforma externa (como o OSF). Uma versão em inglês entra como apêndice do material suplementar do artigo.

---

## 0. Para que serve este documento e como lê-lo

Este **plano de análise** descreve, **antes** de rodar os experimentos de teste, o que esperamos encontrar, como vamos medir e como vamos decidir se acertamos. Depois de fixado, ele não pode mais ser ajustado aos resultados. Isso protege o trabalho de uma crítica comum: a de que os autores testaram muitas coisas e relataram só o que deu certo.

**Por que "plano de análise" e não "pré-registro".** Um pré-registro formal é depositado numa plataforma externa (como o OSF), que fixa a data de forma que terceiros possam conferir. Optamos por não fazer esse depósito. O plano fica versionado no Git do projeto, e sua anterioridade em relação às execuções de teste se apoia nas tags do repositório (Seção 9). **Limitação assumida:** as datas do Git são registradas pelos próprios autores e, em princípio, podem ser alteradas; portanto, não afirmamos que a anterioridade possa ser verificada de forma independente. O que oferecemos é transparência: o plano completo, o histórico do repositório e as regras de decisão, que tornam visível qualquer desvio.

O documento foi escrito para quem **não acompanhou** o desenvolvimento. A Seção 0.1 explica os termos técnicos e os nomes dos braços; a Seção 7 explica cada cálculo estatístico com exemplos numéricos. Existe uma versão em inglês com o mesmo conteúdo (`ANALYSIS_PLAN_EN.md`).

### 0.1 Glossário

| Termo | Significado |
|---|---|
| **CIL** (*class-incremental learning*) | O modelo aprende classes novas em etapas (**tarefas**). Depois de cada tarefa, deve classificar uma imagem entre **todas** as classes vistas até então, sem ser informado de qual tarefa ela veio. |
| **Sem exemplares** | Nenhuma imagem de tarefas antigas é guardada. Só estatísticas resumidas (médias, covariâncias) podem ser guardadas. |
| **Backbone congelado** | Rede pré-treinada (ResNet-18 ou ViT-B/16) que transforma cada imagem num vetor de números, a **feature** φ. Seus pesos nunca mudam. |
| **φ (feature)** | Vetor de 512 números (ResNet-18) ou 768 números (ViT-B/16) que descreve a imagem. |
| **Adapter** | Pequena rede treinável colocada depois do backbone. No nosso método, cada tarefa ganha um adapter próprio, que é congelado quando a tarefa termina. |
| **Protótipo** | Vetor que representa uma classe: a média das features (já passadas pelo adapter) das imagens de treino dessa classe. |
| **Nota (score)** | O quanto uma imagem "parece" com uma classe. Aqui, é o cosseno do ângulo entre a feature da imagem e o protótipo da classe (vai de −1 a 1). A classe com a maior nota é a predição. |
| **Vazamento (Λ)** | Fração das imagens de teste classificadas numa classe de **outra tarefa**. Exemplo: uma imagem de "maçã" (tarefa 1) classificada como "tulipa" (tarefa 3). |
| **Erro dentro da tarefa (E_WP)** | Fração das imagens classificadas numa classe errada **da tarefa certa**. Exemplo: "maçã" classificada como "laranja", ambas da tarefa 1. |
| **NCM** (*nearest class mean*) | Classificador sem treino: compara a feature φ da imagem com a média de φ de cada classe e escolhe a mais próxima pelo cosseno. Serve de referência. |
| **Padronização das notas** | Correção aplicada depois do treino: subtrai das notas de cada tarefa a média μ_k e divide pelo desvio-padrão σ_k que o adapter dessa tarefa teve nos seus próprios dados de treino. |
| **Recusa aprendida** | Ensinar cada adapter, durante o seu treino, a reconhecer **o que não pertence à sua tarefa** e a dar notas baixas a isso. Sem ela, um adapter só aprende a dizer "isto é da minha tarefa", nunca "isto não é meu". |
| **Pseudo-negativos** | Features sintéticas de classes antigas, sorteadas de gaussianas guardadas, usadas para a recusa aprendida: o adapter novo é treinado para não reivindicá-las. |
| **Banco de gaussianas** | Para cada classe já vista, guarda-se a média e uma covariância simplificada de φ. Isso permite sortear features parecidas com as daquela classe sem guardar imagens. |
| **Posto da covariância** | A covariância de uma classe descreve em que direções as features dela variam e quanto. Guardá-la inteira custa d × d números (768 × 768 ≈ 590 mil no ViT). Com **posto r**, guardamos só as r direções em que a classe mais varia (as r componentes principais) e tratamos as demais como uma variação pequena e igual em todas. Posto 32 = as 32 direções principais. |
| **LVQ** (*learning vector quantization*) | Técnica que mantém "unidades" no espaço de features e as aproxima dos exemplos da sua classe e afasta dos de outras. Fazia parte do método inicial como termo extra da perda; foi desligada no desenvolvimento (Seção 1.2). |
| **Oráculo de tarefa** | Situação hipotética em que alguém nos diz a tarefa de cada imagem de teste, e só precisamos escolher a classe dentro dela. A acurácia nessa situação mede apenas a qualidade dentro da tarefa, sem nenhum vazamento possível. |
| **Braço** | Uma configuração experimental (método ou variante) que é comparada às outras. Os principais estão abaixo; a Seção 4.3 detalha cada um. |
| **Braço baseline** | Um adapter por tarefa, treinado só com as imagens da sua tarefa, sem pseudo-negativos. É o sistema cujo problema estudamos. |
| **Braço gauss** | O baseline mais os pseudo-negativos sorteados ao acaso do banco de gaussianas, isto é, com recusa aprendida. |
| **Braço fecam** | Classificador sem treino que, como o NCM, usa só estatísticas das features, mas mede a distância até cada classe levando em conta a forma da nuvem de pontos dela (distância de Mahalanobis), e não só o ângulo. Método publicado (FeCAM, NeurIPS 2023). |
| **Braço gauss_ca** | Usa o mesmo banco de gaussianas, mas de outro jeito: em vez de treinar adapters, retreina um único classificador sobre todas as classes com features sorteadas do banco ("ca" = *classifier alignment*, alinhamento do classificador). Testa se a recusa aprendida vale mais que esse uso simples das gaussianas. |
| **Braço adapter_compartilhado** | Um único adapter para todas as tarefas, que continua sendo treinado a cada tarefa nova. Imita a situação em que a padronização não funcionou na literatura (Xu et al., 2026). |
| **Seed** | Número que fixa todos os sorteios de uma execução (ordem das classes, inicialização, lotes). Seeds diferentes = repetições independentes do experimento. |
| **pp** | Pontos percentuais. De 40% para 44% são +4 pp. |
| **Acurácia final** | Acerto sobre as imagens de teste de todas as classes, depois da última tarefa. |
| **AAC** | Média das acurácias medidas depois de cada tarefa. |
| **Teste t pareado** | Teste estatístico que verifica se a diferença entre dois braços, medida seed a seed, é consistentemente diferente de zero (Seção 7.3). |
| **Correção de Holm** | Ajuste dos valores p quando se fazem vários testes ao mesmo tempo, para que o acaso não produza "achados" falsos (Seção 7.5). |
| **Correlação de Spearman** | Mede se duas grandezas crescem juntas, olhando só a ordem dos valores (quem é maior que quem), e não os valores em si. Vai de −1 a +1 (Seção 7.6). |
| **Teste de permutação unilateral** | Verifica se uma correlação positiva poderia ter surgido por acaso: embaralha os dados de todas as formas possíveis e conta com que frequência aparece uma correlação tão alta quanto a observada. "Unilateral" porque só interessa a direção positiva (Seção 7.6). |
| **Transformação de Tukey** | Elevar cada valor da feature a uma potência (aqui 0,5, a raiz quadrada) para deixar sua distribuição mais simétrica, parecida com uma gaussiana. Só faz sentido para valores não negativos, como os da ResNet. Usada pelo FeCAM. |

---

## 1. O que já foi feito antes deste plano

Este plano é escrito **depois** de uma fase de desenvolvimento. Para que o leitor saiba exatamente o que foi visto e decidido antes, ela é descrita aqui em detalhe.

### 1.1 Cenário usado no desenvolvimento

Todo o desenvolvimento usou **um único cenário**:

| Item | Valor |
|---|---|
| Dataset | CIFAR-100 (100 classes de objetos e animais, 500 imagens de treino e 100 de teste por classe) |
| Divisão em tarefas | 5 tarefas de 20 classes (protocolo "B0 Inc20"), ordem das classes sorteada pela seed |
| Backbone | ResNet-18 pré-treinada no ImageNet, congelada |
| Tamanho da imagem | 64×64 pixels |
| Seeds | 1993 a 1998 |

### 1.2 Linha do tempo das decisões

As decisões abaixo foram tomadas **olhando resultados de teste das seeds 1993 a 1996**. Por isso essas quatro seeds são chamadas de **seeds de desenvolvimento** e não contam como teste.

1. **Ponto de partida.** Adapters por tarefa sobre a ResNet-18 congelada, classificação por cosseno a protótipos, um termo de LVQ (quantização vetorial) na perda e parada antecipada do treino.
2. **Diagnóstico do vazamento.** Ao separar o erro em vazamento e erro dentro da tarefa, viu-se que quase todo o erro era vazamento, e que ele ia quase sempre para as tarefas mais novas. Comparando com o NCM, o vazamento dos adapters era bem maior. Isso definiu o problema estudado.
3. **Correções no classificador.**
    - **Escala s = 16 nos logits.** Cossenos variam só entre −1 e 1; sem multiplicá-los por uma escala, a função softmax da perda não consegue ficar confiante e o treino estagna.
    - **Protótipos novos iniciados na média de φ e recalculados a cada época.** Antes, os protótipos usados no treino e na avaliação eram diferentes; com a mudança, treino e avaliação usam os mesmos alvos.
    - **LVQ desligado** (pesos α_lvq = 0,4 e λ_lvq = 0,3 passaram a 0). As unidades da LVQ começavam na origem e nunca se moviam durante o treino, então o termo não contribuía para o aprendizado.
4. **Pseudo-negativos.** Para atacar o vazamento no treino, foi criado o banco de gaussianas e a perda de recusa. Três formas de escolher os pseudo-negativos foram implementadas: sorteio direto (gauss), seleção pela margem de vazamento (margem) e, mais tarde, margem calculada sobre notas padronizadas (margem_padronizada).
5. **Amostradores aprendidos (GFlowNet).** Foi considerado treinar uma GFlowNet para sortear pseudo-negativos. Um estudo controlado em 2 dimensões mostrou que a amostragem por importância simples já reproduzia a distribuição desejada, e que a GFlowNet só se igualava a ela quando treinada com as próprias amostras da amostragem por importância. A ideia foi abandonada.
6. **Parada antecipada, primeira versão (seed 1993).** Parar quando a perda de validação das classes novas parasse de cair interrompia o treino **antes** de o adapter aprender a recusar classes antigas.
7. **Parada antecipada, segunda versão (seeds 1993–1996).** Acompanhando as duas perdas (classes novas e recusa), a acurácia passou a se correlacionar com o número de épocas treinadas (correlação de Pearson 0,85). Ou seja, a parada antecipada só atrapalhava. **Decisão:** treino fixo de 100 épocas por tarefa.
8. **Padronização das notas.** Um diagnóstico (a parcela de escala, κ) indicou que parte do vazamento vinha de os adapters terem notas em escalas diferentes. Aplicada depois do treino aos checkpoints das seeds 1993–1996 (treinados com parada antecipada), a padronização levou a acurácia final do baseline de 24,0% para 40,6%. **Decisão:** toda avaliação passa a ser feita com notas brutas e com notas padronizadas.
9. **Braço margem_padronizada.** Criado para que a seleção pela margem use as mesmas notas (padronizadas) que decidem a classe no teste.
10. **Aceleração.** As features φ passaram a ser calculadas uma única vez e guardadas em disco. Como não há aumento de dados, isso não muda nenhum resultado, só o tempo.
11. **Código final da fase de desenvolvimento** rodado nas seeds 1993 a 1998, em 18 e 19/11/2026. O código e todos os resultados da fase de desenvolvimento estão marcados com a tag git `v0.9-desenvolvimento` no repositório `eavcampos/CIL_e_GFlowNet` (commit `ec84535484056e6d666ad0b2708a4376a06fbbb4`, de 23/11/2026). Isso inclui as pastas `resultados_parada_val_ce/`, `resultados_parada_ambas/` e `resultados_100epocas_v1/`, das execuções com parada antecipada (itens 6 e 7).

### 1.3 Situação de cada seed e de cada cenário

| Item | Situação |
|---|---|
| Seeds 1993–1996 | **Desenvolvimento.** Orientaram as decisões 2 a 9. |
| Seeds 1997 e 1998 | Rodadas com o código final e analisadas em 19/11/2026. **Nenhuma decisão foi tomada depois de vê-las.** São tratadas como teste já observado (Seção 7.9). |
| Seeds 1999–2002 | Nunca rodadas. |
| ViT-B/16; ImageNet-R, ImageNet-A, OmniBenchmark, VTAB; CIFAR-100 com 20 tarefas | Nunca rodados. |
| Braços FeCAM, gauss_ca, adapter_compartilhado | Ainda não implementados. |

### 1.4 Resultados do desenvolvimento que motivam as previsões

CIFAR-100, 5 tarefas, ResNet-18, média de seis seeds (1993–1998):

| Braço | Acurácia final, notas brutas | Acurácia final, notas padronizadas | Vazamento Λ, brutas | Vazamento Λ, padronizadas | Erro dentro da tarefa, padronizadas |
|---|---|---|---|---|---|
| NCM | 41,3% | — (não se aplica) | 46,7% | — | 12,0% |
| Baseline | 24,6% | 39,4% | 68,5% | 53,6% | 7,0% |
| Gauss | 34,3% | 43,4% | 58,0% | 48,5% | 8,1% |
| Margem_padronizada | 35,3% | 44,1% | 56,8% | 47,4% | 8,5% |

Lembrete: para cada imagem, acerto + vazamento + erro dentro da tarefa = 100%. Por exemplo, Gauss padronizado: 43,4% + 48,5% + 8,1% = 100%.

**O que cada braço ganha ou perde.** Todas as diferenças abaixo são médias pareadas nas seis seeds; "6/6" quer dizer que a diferença teve o mesmo sinal em todas as seeds.

- **NCM (referência, sem treino):** 41,3%. Erra 58,7% das imagens: 46,7 pp por vazamento e 12,0 pp dentro da tarefa. Esse vazamento é o "normal", causado só por classes parecidas em tarefas diferentes.
- **Baseline com notas brutas: −16,7 pp em relação ao NCM** (24,6% contra 41,3%; 6/6). Os adapters são **melhores** que o NCM dentro de cada tarefa (erro de 7,0% contra 12,0%; com oráculo de tarefa, 70,9% contra 65,2%), mas vazam 21,8 pp a mais, quase sempre para as tarefas mais novas. O ganho dentro da tarefa é engolido pelo vazamento.
- **Padronizar o baseline: +14,8 pp** (de 24,6% para 39,4%; 6/6), sem retreinar nada. O vazamento cai de 68,5% para 53,6% e o erro dentro da tarefa não muda, como esperado (Seção 1.5). Mesmo assim, o baseline padronizado fica **1,9 pp abaixo do NCM**: ainda vaza 6,9 pp a mais que ele.
- **Pseudo-negativos (gauss) com notas brutas: +9,7 pp em relação ao baseline bruto** (34,3% contra 24,6%), mas ainda 7,0 pp abaixo do NCM.
- **Pseudo-negativos depois da padronização: +4,0 pp em relação ao baseline padronizado** (43,4% contra 39,4%; 6/6). O vazamento cai mais 5,1 pp; o erro dentro da tarefa sobe 1,2 pp, que é o preço de o adapter dedicar parte da sua capacidade a recusar classes antigas.
- **Combinação (gauss padronizado): +2,1 pp em relação ao NCM** (6/6). É a única configuração, junto com as de margem, que supera o NCM: vaza quase o mesmo que ele (48,5% contra 46,7%) e erra menos dentro da tarefa (8,1% contra 12,0%).
- **Seleção pela margem (margem_padronizada) em relação ao sorteio (gauss): +0,7 pp**, diferença **não significativa** (intervalo de 95% de −0,2 a +1,6; Seção 7.3). Contra o NCM, +2,8 pp (6/6).

Nenhum dos dois mecanismos, sozinho, supera o NCM; juntos, superam. Esses resultados são de **um único cenário** e de seeds que, em parte, orientaram o desenvolvimento. É por isso que precisam ser testados.

### 1.5 Por que os adapters dão notas em escalas diferentes, e como isso foi percebido

**O que significa "escalas diferentes".** Cada adapter dá notas (cossenos) às classes da sua tarefa. Se o adapter da tarefa 4 costuma dar notas em torno de 0,8 a qualquer imagem, e o da tarefa 0 em torno de 0,5, o da tarefa 4 vence quase sempre, mesmo em imagens da tarefa 0. A comparação entre tarefas fica injusta, como comparar notas de duas provas corrigidas com rigores diferentes.

**Por que isso acontece.** Três razões, que se somam:
1. **Nada fixa o nível das notas.** Cada adapter é treinado com entropia cruzada apenas entre as classes da sua própria tarefa. Essa perda só depende de **quanto a classe certa supera as outras classes da mesma tarefa**, e não do valor absoluto das notas. Somar a mesma quantidade a todas as notas de uma tarefa não muda a perda. Por isso, o nível em que as notas de cada adapter terminam é arbitrário, e nada obriga os adapters a terminarem no mesmo nível.
2. **As tarefas têm dificuldades diferentes.** Numa tarefa com classes bem separadas, o adapter aprende a dar notas altas com facilidade; numa com classes parecidas, as notas típicas ficam mais baixas.
3. **Nenhum adapter aprende a recusar.** O adapter é treinado para puxar as imagens da sua tarefa para perto dos seus protótipos. Como nunca vê imagens de outras tarefas, ele também puxa essas imagens para perto, sem nenhuma penalidade. O quanto isso acontece varia de adapter para adapter.

A **direção** observada (os adapters mais novos vencem) foi medida, mas a sua causa exata **não foi isolada** no desenvolvimento. É uma das coisas que P1 e o registro das estatísticas μ_k de cada tarefa (Seção 6) vão esclarecer.

**Como o fenômeno foi percebido.** Por três evidências, nesta ordem:
1. **A matriz de vazamento** mostrou que as imagens que saíam da sua tarefa iam quase todas para as tarefas mais recentes: 76% das imagens da primeira tarefa eram atribuídas à última, e a acurácia final por tarefa ia de 0,1% (tarefa 0) a 62,7% (tarefa 4). No NCM, que não tem adapters, os erros se espalhavam por igual entre as tarefas.
2. **O teste da padronização.** A padronização subtrai uma constante e divide por outra **dentro de cada tarefa**. Por isso ela não pode mudar qual classe vence dentro de uma tarefa; só pode mudar qual tarefa vence. Se o problema fosse só sobreposição real, ela não ajudaria. Aplicada ao **mesmo** modelo treinado, ela deu +14,8 pp, e o índice de direção caiu de 31,2 para 1,7 (1 = sem preferência). Isso só é possível se boa parte do vazamento vier de diferenças de nível e espalhamento entre as notas das tarefas.
3. **A parcela de escala κ = 0,22:** a padronização eliminou cerca de um quinto do vazamento do baseline, justamente a parte em excesso em relação ao NCM, que é a mais fácil de recuperar.

**O que ainda não foi medido diretamente:** os valores de μ_k e σ_k de cada tarefa não foram gravados nas execuções de desenvolvimento. Nas execuções de teste eles passam a ser gravados (Seção 6), o que permite ver as diferenças de escala diretamente e calcular a dispersão de escala DE.

---

## 2. Objetivos

### 2.1 Objetivo geral

Determinar **quando** um sistema de CIL com adapters por tarefa sobre um backbone congelado precisa apenas de **calibração das notas** (padronização, aplicada depois do treino) e quando precisa também que **cada adapter aprenda, já no treino, a reconhecer o que não pertence à sua tarefa** (recusa aprendida, com pseudo-negativos), usando indicadores que podem ser medidos **antes** de aplicar qualquer correção.

Em termos simples: um adapter que vence onde não deveria pode estar só "falando mais alto" que os outros (problema de escala, corrigível depois, quase de graça) ou pode realmente ter aprendido a reivindicar o que não é seu (problema de sobreposição, que só um treino diferente resolve). Queremos saber distinguir os dois casos e saber quando cada remédio é necessário.

### 2.2 Por que isso importa

Métodos recentes de CIL com modelos pré-treinados (EASE, MOS, TUNA) usam um módulo por tarefa e gastam esforço considerável para escolher o módulo certo no teste. Saber se o problema é de escala (corrigível depois do treino, a custo quase zero) ou de sobreposição real (que exige mudar o treino) indica qual tipo de solução usar. Além disso, a mesma padronização foi testada por Xu et al. (2026) com um encoder compartilhado e não trouxe ganho; nossos dados de desenvolvimento mostram ganho grande com adapters independentes. Um diagnóstico que explique essa diferença é útil para a área.

### 2.3 Objetivos específicos

| # | Objetivo específico | Previsão que o testa |
|---|---|---|
| OE1 | Verificar se o excesso de vazamento direcionado às tarefas novas, visto no desenvolvimento, aparece em outros datasets, protocolos e backbones | P1 |
| OE2 | Verificar se indicadores medidos sem aplicar a padronização preveem quanto ela vai ajudar | P2 |
| OE3 | Verificar se os pseudo-negativos acrescentam ganho depois da padronização, e se usá-los para treinar adapters é melhor que usá-los para realinhar um classificador único | P3 |
| OE4 | Testar por que a padronização funcionou nos nossos dados e não nos de Xu et al. (2026). Nossa explicação é que as diferenças de escala nascem de cada tarefa ter **seu próprio** adapter, treinado isoladamente. Quando todas as tarefas compartilham **um único** adapter, como em Xu et al., as notas de todas são produzidas pela mesma rede e deveriam ficar em escalas parecidas. Testamos isso com o braço adapter_compartilhado: se a explicação estiver certa, nele a padronização quase não mudará a acurácia | P4 |
| OE5 | Mostrar onde o sistema completo (padronização + pseudo-negativos) se situa em relação a dois grupos: (a) classificadores que não treinam nada e só usam estatísticas das features (NCM e FeCAM), comparação que fazemos nós mesmos, com estatística completa; (b) os métodos publicados mais fortes (RanPAC, EASE, MOS e TUNA), comparação feita inicialmente com os números dos artigos. Não esperamos superar o grupo (b); o objetivo é que o leitor saiba o tamanho da distância e o custo (memória e tempo) de cada abordagem | Questão secundária S1 |

### 2.4 Fora do escopo

Propor um método novo de estado da arte; ajustar hiperparâmetros por dataset; usar exemplares; fazer *fine-tuning* do backbone nos nossos braços.

---

## 3. Previsões

### 3.1 Medidas usadas nas previsões

Todas são calculadas por seed e por cenário. O cálculo de cada uma está detalhado na Seção 7.2.

- **Cenário:** uma combinação dataset × protocolo × backbone (lista na Seção 4.1).
- **Ganho da padronização, Δ_pad:** acurácia final do baseline com notas padronizadas menos a acurácia final do **mesmo** baseline com notas brutas. O modelo é o mesmo; só muda a regra de decisão.
- **Excesso de vazamento:** Λ do baseline (notas brutas) menos Λ do NCM.
- **Índice de direção, D:** quanto do vazamento vai para tarefas mais novas, dividido por quanto vai para tarefas mais antigas. D = 1 significa nenhuma preferência; D = 30 significa que vaza 30 vezes mais para as novas. Calculado só com notas brutas.
- **Dispersão de escala, DE:** o quanto as notas típicas dos adapters diferem entre si, em unidades do espalhamento típico de cada um. Calculada só com dados de treino.

**Por que κ não é usado como preditor.** A parcela de escala κ = (Λ − Λ̃)/Λ usa Λ̃, o vazamento **com notas padronizadas**. Ou seja, para calcular κ é preciso já ter aplicado a padronização. Correlacionar κ com o ganho da padronização seria medir quase a mesma coisa duas vezes e encontrar uma correlação "de graça". D e DE não dependem da padronização e, por isso, servem como preditores honestos. κ continua sendo reportado, apenas como descrição.

### 3.2 P1 — Excesso de vazamento direcionado às tarefas novas

**Em palavras simples:** quando cada tarefa tem seu próprio adapter, treinado sem nunca ver as outras, os adapters mais novos "roubam" imagens das tarefas antigas, mais do que um classificador sem treino (NCM) confundiria por acaso.

**Por que esperamos isso:** cada adapter é treinado para dar notas altas às suas classes e nunca é cobrado por dar notas altas a imagens de outras tarefas. No desenvolvimento, o excesso foi de 21,8 pp e D = 31,2.

**Como medimos, em cada cenário:**
(a) a diferença de vazamento entre o baseline (notas brutas) e o NCM, seed a seed, com teste t pareado (Seção 7.4);
(b) o índice de direção D do baseline, pela mediana das seeds.

**Confirmada no cenário** se (a) for positiva e significativa depois da correção de Holm entre os cenários (Seção 7.5) **e** a mediana de D for maior que 1.
**Refutada no cenário** se a diferença média de (a) for menor ou igual a zero, ou se a mediana de D for menor ou igual a 1.
**Inconclusiva no cenário** nos demais casos (diferença positiva mas não significativa, por exemplo).
**Resultado global:** confirmada se confirmada em todos os 8 cenários; parcialmente confirmada se em 5 ou mais; refutada caso contrário.

**O que cada desfecho significa:** se P1 falhar com o ViT, o problema estudado é próprio de features fracas, e o artigo diz isso.

### 3.3 P2 — Os indicadores de escala preveem o ganho da padronização

**Em palavras simples:** nos cenários em que os adapters "falam em volumes diferentes" (DE alto) ou em que o vazamento é muito desequilibrado para as tarefas novas (D alto), a padronização ajuda mais.

**Por que esperamos isso:** a padronização só corrige diferenças de escala. Se as escalas já forem parecidas, ela não tem o que corrigir.

**Como medimos:** cada um dos 8 cenários vira **um ponto**. Para cada ponto calculamos a média de Δ_pad nas seis seeds, a mediana de D e a média de DE. Depois calculamos a correlação de Spearman (Seção 7.6) entre Δ_pad e D, e entre Δ_pad e DE.

**Confirmada** se pelo menos uma das duas correlações for positiva com p < 0,05 no teste de permutação unilateral, depois da correção de Holm para 2 testes.
**Refutada** se as duas correlações forem menores ou iguais a zero.
**Inconclusiva** nos demais casos.

**O que cada desfecho significa:** se P2 for confirmada, o diagnóstico vira uma ferramenta: dá para saber, antes de corrigir, se vale a pena padronizar. Se for refutada, o artigo reporta o efeito da padronização sem afirmar que ele é previsível.

### 3.4 P3 — A recusa acrescenta ganho depois da padronização e supera o realinhamento do classificador

**Em palavras simples:** (a) mesmo depois de padronizar, ensinar o adapter novo a recusar features sintéticas de classes antigas melhora a acurácia; (b) usar essas features para treinar adapters é melhor do que usá-las do jeito mais simples, que é retreinar um classificador único sobre todas as classes (braço gauss_ca).

**Por que esperamos isso:** a padronização corrige a escala, mas não muda **onde** o adapter coloca as imagens. Se um adapter novo põe imagens antigas perto dos seus protótipos, só o treino resolve. No desenvolvimento, (a) deu +4,0 pp; (b) nunca foi testada.

**Como medimos, em cada cenário:**
(a) acurácia final de gauss (padronizado) menos baseline (padronizado), seed a seed, teste t pareado;
(b) acurácia final de gauss (padronizado) menos gauss_ca, seed a seed, teste t pareado.
A correção de Holm é aplicada entre os 8 cenários, separadamente para (a) e para (b).

**Confirmada no cenário** se (a) e (b) forem positivas e significativas. **Resultado global** com a mesma regra de P1.
**Desfecho registrado de antemão:** se (a) valer e (b) falhar, a conclusão será "os pseudo-negativos ajudam, mas não há vantagem em usá-los para treinar adapters em vez de realinhar o classificador". Nesse caso, a recusa sai da lista de contribuições do artigo.

### 3.5 P4 — Com encoder compartilhado, a escala importa pouco

**Em palavras simples:** se todas as tarefas usarem **o mesmo** adapter, atualizado a cada tarefa, as notas das tarefas ficam em escalas parecidas, e a padronização quase não muda nada. Isso explicaria por que Xu et al. (2026) não viram ganho.

**Como medimos, nos 4 cenários onde o braço roda:**
(a) Δ_pad do adapter compartilhado menos Δ_pad do baseline, seed a seed, teste t pareado; espera-se diferença negativa;
(b) a média de Δ_pad do adapter compartilhado.

**Confirmada** se, em todos os 4 cenários, (a) for negativa e significativa depois de Holm **e** o valor absoluto de (b) for menor que 1 pp.
**Refutada** se, em algum cenário, (a) for maior ou igual a zero.
**Inconclusiva** nos demais casos.

### 3.6 Questões secundárias (sem previsão de direção)

- **S1.** Como a combinação padronização + pseudo-negativos (braço gauss, notas padronizadas) se compara a métodos sem treino e a métodos publicados? Sem previsão de direção. Tem duas partes:
    - **S1a — contra NCM e FeCAM**, que rodamos nós mesmos: comparação pareada com as estatísticas da Seção 7, em todos os cenários.
    - **S1b — contra RanPAC, EASE, MOS e TUNA**: na fase inicial, usamos os números publicados (Seção 4.3.7 e Anexo A). Como cada artigo traz um único número, de uma única execução com a seed 1993, não há teste estatístico possível: a comparação é **descritiva**, feita contra os nossos braços rodados na mesma seed 1993 (Seção 4.2). Se depois rodarmos esses métodos nas seeds de teste, a comparação passa a seguir a Seção 7, e o artigo indica de onde vem cada número.
- **S2.** Selecionar os pseudo-negativos pela margem (braços margem e margem_padronizada) muda o resultado em relação ao sorteio direto (gauss)? No desenvolvimento não houve diferença significativa. Esperamos que continue assim, mas isso não é uma previsão.

---

## 4. Desenho experimental

### 4.1 Cenários

| # | Dataset | Classes | Protocolo | Tarefas | Backbone | Entrada | Usado em |
|---|---|---|---|---|---|---|---|
| 1 | CIFAR-100 | 100 | B0 Inc20 | 5 | ResNet-18 | 64×64 | P1–P4 |
| 2 | ImageNet-R | 200 | B0 Inc20 | 10 | ResNet-18 | 64×64 | P1–P4 |
| 3 | CIFAR-100 | 100 | B0 Inc20 | 5 | ViT-B/16 | 224×224 | P1–P4 |
| 4 | CIFAR-100 | 100 | B0 Inc5 | 20 | ViT-B/16 | 224×224 | P1–P3, ablações |
| 5 | ImageNet-R | 200 | B0 Inc20 | 10 | ViT-B/16 | 224×224 | P1–P4, ablações |
| 6 | ImageNet-A | 200 | B0 Inc20 | 10 | ViT-B/16 | 224×224 | P1–P3 |
| 7 | OmniBenchmark | 300 | B0 Inc30 | 10 | ViT-B/16 | 224×224 | P1–P3 |
| 8 | VTAB | 50 | B0 Inc10 | 5 | ViT-B/16 | 224×224 | P1–P3 |

**"B0 IncX"** quer dizer: nenhuma classe pré-aprendida antes da primeira tarefa (B0) e X classes novas por tarefa.

**Por que esses cenários:**
- **Cenário 1** é o do desenvolvimento; permite verificar se os resultados se repetem com seeds novas.
- **Cenários 1, 2 e 3** permitem comparar backbones no mesmo dataset e protocolo (1 contra 3) e o mesmo backbone fraco em outro dataset (1 contra 2). Isso cria variação de escala entre cenários, necessária para P2.
- **Cenários 4 a 8** usam exatamente os protocolos do artigo do MOS (Sun et al., AAAI 2025); os cenários 4, 5 e 6 também coincidem com os do TUNA (Wang et al., ICCV 2025). Isso permite comparar nossos números com os publicados. Para ImageNet-R, o artigo do EASE usa Inc5; adotamos Inc20, que é o do MOS, o do TUNA e o do arquivo `exps/ease_inr.json` do LAMDA-PILOT.
- **Cenários 3 e 4** diferem só no número de tarefas (5 contra 20), o que mostra o efeito de ter mais adapters competindo.

**Dados:** as cópias de cada dataset distribuídas no README do LAMDA-PILOT, já divididas em treino e teste, que são as usadas por SimpleCIL, EASE, MOS e TUNA.

**Por que o ObjectNet foi excluído.** A literatura de CIL com modelos pré-treinados costuma incluir um benchmark com 200 classes do ObjectNet, dividido em treino e teste. A licença do ObjectNet, porém, diz: *"ObjectNet may never be used to tune the parameters of any model"* (o ObjectNet nunca pode ser usado para ajustar os parâmetros de nenhum modelo). Treinar adapters ou calcular protótipos com essas imagens viola essa cláusula. Por isso o ObjectNet não é usado em nenhum cenário. **Licenças dos datasets usados (conferidas em 22 e 23/11/2026).** Nenhuma delas proíbe treinar modelos:

| Dataset | Licença ou termos encontrados |
|---|---|
| CIFAR-100 | sem licença formal; uso acadêmico difundido |
| ImageNet-A, ImageNet-R | MIT (repositórios dos autores) |
| OmniBenchmark | Creative Commons BY-NC-ND 4.0: uso não comercial e sem distribuição de versões modificadas. O nosso uso (pesquisa, sem redistribuir os dados) é compatível |
| VTAB, conjunto de 50 classes do APER | formado por 10 classes de cada um de 5 datasets. São RESISC45, DTD, Oxford-IIIT Pets, EuroSAT e Flowers-102, nessa ordem (pastas 10–19, 20–29, 30–39, 40–49 e 50–59). As pastas são numeradas, sem nomes; a composição foi confirmada em 23/11/2026 pelas contagens da cópia local, que batem com a versão do PILOT (1.796 imagens de treino e 8.619 de teste) e com o protocolo VTAB-1k (cerca de 1.000 imagens de treino por dataset de origem): 40 imagens de teste por classe = divisão oficial do DTD; ~100 de treino por classe e 5.400 de teste = EuroSAT; ~9 de treino por classe = Flowers-102. EuroSAT: MIT. Oxford-IIIT Pets: Creative Commons (BY 4.0 numa cópia no Zenodo; conferir a página oficial). DTD, Flowers-102 e RESISC45: sem licença explícita nas cópias consultadas. O próprio VTAB foi criado para treinar e avaliar modelos nesses datasets |

Dois cuidados: (i) o autor do APER, que montou os subconjuntos de OmniBenchmark e VTAB distribuídos pelo PILOT, avisa no README que não tem direito de redistribuí-los; por isso **não redistribuímos dados** — o código só aponta para os links originais; (ii) o artigo cita a licença de cada dataset.

**Ordem das classes:** `numpy.random.seed(seed)` seguido de `numpy.random.permutation(C)`, como no LAMDA-PILOT. O nosso código usa `RandomState(seed).choice(C, C, replace=False)`; foi verificado em 22/11/2026 que as duas formas geram exatamente a mesma ordem para C = 50, 100, 200 e 300. Os índices de classe seguem a ordem alfabética das pastas, também como no PILOT. **Exceção: VTAB.** Os autores do benchmark determinam que, no VTAB, as classes **não** sejam embaralhadas (`"shuffle": false`), para que cada tarefa corresponda a um dos 5 datasets de origem (README do repositório do APER e arquivo `exps/simplecil/vtab_10_10_vit-b_simplecil.json`). Seguimos isso: no cenário 8, a ordem das classes é a mesma em todas as seeds, e as seeds variam só os sorteios do treino (inicialização, lotes, pseudo-negativos). Como as pastas se chamam 10 a 59 (sempre dois dígitos), a ordem alfabética coincide com a numérica, e cada tarefa recebe exatamente um dataset de origem.

**Duas particularidades do VTAB, registradas de antemão:** (i) o teste é muito desequilibrado entre tarefas: a tarefa do EuroSAT tem 5.400 das 8.619 imagens de teste (63%), então a acurácia final e o vazamento são dominados por ela; por isso o VTAB é reportado também com a acurácia por tarefa; (ii) a tarefa do Flowers-102 tem de 7 a 11 imagens de treino por classe, menos que o posto 32 do banco de gaussianas: a covariância dessas classes fica limitada a poucas direções e depende do encolhimento. Isso é uma limitação do cenário, não um motivo para mudar hiperparâmetros.

### 4.2 Seeds

- **Teste:** 1997, 1998, 1999, 2000, 2001 e 2002, em todos os cenários e braços, inclusive nas ablações.
- **Seed 1993, só para a comparação com números publicados (S1b):** os braços ncm, fecam, gauss_ca, baseline e gauss também rodam com a seed 1993 nos cenários 4 a 8. Os números publicados vêm de uma execução com essa seed, então a comparação é feita com a **mesma ordem de classes**. Esses resultados não entram em nenhuma previsão. A seed 1993 só foi seed de desenvolvimento no cenário 1 (CIFAR-100, ResNet-18); com o ViT e nos outros datasets, ela nunca foi vista.
- **RanPAC, EASE, MOS e TUNA, se rodados por nós mais tarde:** as mesmas seis seeds de teste. Se o tempo de GPU não permitir, 1997, 1998 e 1999, e isso é declarado no artigo.

### 4.3 Braços

Cada braço treinado é avaliado **duas vezes**, com notas brutas e com notas padronizadas, sem retreinar. Os braços ncm, fecam e gauss_ca, que não têm adapters por tarefa, são avaliados só da forma original, porque todas as suas classes já compartilham uma mesma escala. Os métodos publicados (Seção 4.3.7) entram, na fase inicial, pelos números dos artigos.

#### 4.3.1 ncm — classificador pela média, sem treino
- **O que guarda:** a média de φ de cada classe (512 ou 768 números por classe).
- **Como classifica:** calcula o cosseno entre φ da imagem e a média de cada classe; escolhe a maior.
- **O que treina:** nada.
- **Por que está aqui:** é a referência que diz quanto vazamento é "normal". Como todas as classes vivem no mesmo espaço, não há adapters competindo. É idêntico ao SimpleCIL (Zhou et al., IJCV 2024).
- **Serve a:** P1 (referência do excesso), S1 e a verificação de protocolo (Seção 8).

#### 4.3.2 fecam — classificador por distância de Mahalanobis, sem treino
- **O que guarda:** média e matriz de covariância completa de φ para cada classe.
- **Como classifica:** usa a distância de Mahalanobis, que mede a distância até a média de cada classe levando em conta o formato da nuvem de pontos daquela classe (direções em que a classe varia muito pesam menos). Escolhe a classe mais próxima.
- **O que treina:** nada.
- **Implementação:** segue o repositório oficial (Goswami et al., NeurIPS 2023, `dipamgoswami/FeCAM`, conferido em 22/11/2026), versão sem treino. Antes da inversão, a covariância de cada classe recebe um **encolhimento**: soma-se α1 × (média da diagonal) à diagonal e α2 × (média fora da diagonal) fora dela, o que evita uma matriz quase singular quando há poucas imagens por classe. Depois, ela é **normalizada** para matriz de correlação. Valores, copiados dos scripts e configurações oficiais:
    - ViT-B/16, CIFAR-100: α1 = 1, α2 = 0, sem transformação de Tukey (script `FeCAM_vit_cifar100.py`);
    - ViT-B/16, demais datasets: α1 = 10, α2 = 10, sem transformação de Tukey (script `FeCAM_vit_imagenet-r.py`, o único outro script com ViT em imagens naturais);
    - ResNet-18: α1 = 1, α2 = 1, com transformação de Tukey β = 0,5, que eleva cada valor da feature à potência 0,5 para deixar a distribuição mais próxima de uma gaussiana (`exps/FeCAM_cifar100.json`).
- **Por que está aqui:** guarda informação comparável ao nosso banco de gaussianas (média e covariância por classe). Sem ele, um revisor poderia dizer que nossa vantagem vem só de guardar covariâncias.
- **Serve a:** S1.

#### 4.3.3 gauss_ca — classificador único realinhado com features sintéticas
- **O que guarda:** o mesmo banco de gaussianas dos nossos braços (Seção 5.1).
- **Como classifica:** um classificador de cosseno com um vetor por classe, sobre φ, sem adapters.
- **O que treina:** depois de cada tarefa, só o classificador, com features sorteadas do banco para **todas** as classes vistas.
- **Detalhes fixados:** vetores iniciados nas médias de φ; depois de cada tarefa, 5 épocas; em cada época, 256 features sorteadas por classe vista; SGD com taxa 0,01 e momentum 0,9; lote 48; escala s = 16; ReLU nas amostras só na ResNet-18.
- **Por que está aqui:** é a forma mais simples e conhecida de usar features sintéticas (é a etapa de "alinhamento do classificador" do SLCA, Zhang et al., ICCV 2023). Se ela for tão boa quanto treinar adapters para recusar, a recusa não se justifica.
- **Serve a:** P3(b) e S1.

#### 4.3.4 baseline — adapters por tarefa sem pseudo-negativos
- **O que guarda:** o adapter de cada tarefa (congelado ao fim dela) e o protótipo de cada classe.
- **Como classifica:** a imagem passa pelo backbone e por **todos** os adapters; cada adapter dá notas às classes da sua tarefa (cosseno ao protótipo); vence a maior nota entre todas as classes.
- **O que treina:** em cada tarefa, só o adapter novo, com entropia cruzada nas imagens da tarefa atual.
- **Por que está aqui:** é o sistema cujo problema estudamos.
- **Serve a:** P1, P2, P3(a), P4.

#### 4.3.5 gauss — baseline com pseudo-negativos sorteados
- **O que guarda:** o mesmo do baseline, mais o banco de gaussianas.
- **O que treina:** como o baseline, mas cada lote recebe também 48 features sintéticas de classes antigas, sorteadas ao acaso do banco e rotuladas com sua classe antiga. Como os adapters e protótipos antigos estão congelados, a única forma de acertar essas features é o adapter novo dar notas baixas a elas, isto é, aprender a recusá-las.
- **Por que está aqui:** é o nosso mecanismo de recusa na forma mais simples.
- **Serve a:** P3 e S1.

#### 4.3.6 adapter_compartilhado — um único adapter para todas as tarefas
- **O que guarda:** um único adapter e os protótipos de todas as classes.
- **O que treina:** o mesmo adapter continua sendo treinado a cada tarefa, com entropia cruzada nas classes da tarefa atual. Os protótipos de classes antigas ficam como estavam ao fim da sua tarefa.
- **Padronização:** μ_k e σ_k calculados ao fim de cada tarefa k, com o adapter naquele momento.
- **Por que está aqui:** imita a situação de Xu et al. (2026), em que todas as tarefas compartilham o mesmo extrator de features. É o teste direto da nossa explicação.
- **Serve a:** P4. Roda nos cenários 1, 2, 3 e 5.

#### 4.3.7 Métodos publicados: RanPAC, EASE, MOS e TUNA
- **O que são:**
    - **RanPAC** (McDonnell et al., NeurIPS 2023): projeta as features do backbone para um espaço de dimensão muito maior com uma matriz aleatória fixa e uma não linearidade, e classifica por regressão de cumeeira (*ridge*) sobre protótipos de classe, descorrelacionados pela matriz de Gram. Na versão padrão, ajusta um adapter só na primeira tarefa. Não usa adapters por tarefa, mas é um dos métodos mais fortes da área e quase não treina.
    - **EASE** (Zhou et al., CVPR 2024): treina adapters **dentro** das camadas do ViT, um conjunto por tarefa, e sintetiza protótipos de classes antigas nos subespaços novos.
    - **MOS** (Sun et al., AAAI 2025): também usa adapters por tarefa no ViT, funde adapters para reduzir o esquecimento e escolhe o adapter de cada imagem por um mecanismo de recuperação sem treino.
    - **TUNA** (Wang et al., ICCV 2025): combina adapters por tarefa com um adapter universal; no teste, escolhe o adapter de tarefa pela entropia das predições e soma as predições dele às do adapter universal.
- **Como entram, na fase inicial:** pelos **números publicados**, nos cenários 4 a 8, desde que a fonte use o mesmo protocolo: ViT-B/16-IN21K, a mesma divisão em tarefas e a ordem de classes da seed 1993. O Anexo A lista a fonte (artigo e tabela) de cada número. No artigo, cada número copiado é marcado como "reportado em [referência]". Célula sem fonte com protocolo idêntico fica vazia ("—"); nunca se usa número de outro protocolo.
- **Por que é legítimo copiar:** a verificação da Seção 8.1 testa se o nosso pipeline reproduz o SimpleCIL publicado. Se reproduzir, os nossos dados, transformações e checkpoint são os mesmos dos artigos, e os números são comparáveis. A verificação da Seção 8.2 reproduz um dos quatro métodos para confirmar isso.
- **Se rodarmos depois:** código oficial do LAMDA-PILOT, sem alterações. Hiperparâmetros: os dos arquivos de configuração do PILOT, onde existirem (há para CIFAR-100 e ImageNet-R); para os outros datasets, os informados nos artigos originais ou no material suplementar deles; onde não houver informação, os do ImageNet-R. A escolha final por dataset é anotada em adendo antes de rodar.
- **Por que estão aqui:** são o estado da arte em CIL com modelos pré-treinados e sem exemplares. EASE, MOS e TUNA usam adapters por tarefa, como nós, e tratam o mesmo problema de escolher o módulo certo; o RanPAC é a referência forte que quase não treina. Sem eles, não há como posicionar o trabalho.
- **Serve a:** S1b. Só com ViT. O cenário 3 (CIFAR-100 com 5 tarefas, ViT) não tem números publicados.

#### 4.3.8 margem e margem_padronizada — seleção dos pseudo-negativos pela margem
Descritos na Seção 4.4 (ablação A4). Rodam nos cenários 4 e 5.

### 4.4 Ablações

Uma **ablação** muda uma única peça do método e observa o efeito. As ablações abaixo são **exploratórias**: não têm previsão e não decidem nenhuma conclusão principal, mas ajudam a entender por que o método funciona. Todas rodam nos cenários 4 e 5 (ViT, CIFAR-100 com 20 tarefas e ImageNet-R), com as seis seeds de teste.

**A1. De onde vêm as estatísticas da padronização.**
- **Pergunta:** a padronização funcionaria melhor se μ_k e σ_k fossem calculados em imagens que o adapter não viu no treino?
- **Por que perguntar:** depois de 100 épocas, o adapter dá notas muito confiantes às imagens de treino. As estatísticas calculadas nelas podem não representar as notas de imagens novas.
- **Como:** o baseline é treinado em 90% das imagens de treino de cada classe; os 10% restantes ficam de fora. O **mesmo** modelo é avaliado de duas formas: com μ_k, σ_k calculados nos 90% usados no treino, e com μ_k, σ_k calculados nos 10% não vistos.
- **Valores:** {estatísticas do treino, estatísticas da validação}.
- **Leitura:** se a versão com validação for claramente melhor, parte do vazamento que sobra depois da padronização é calibração ruim, e não sobreposição real.

**A2. Peso da perda de recusa, λ_pn.**
- **Pergunta:** quanto peso a recusa deve ter em relação à classificação das classes novas?
- **Como:** braço gauss com λ_pn em {0,25; 0,5; 1,0}. O valor principal é 0,5.
- **Leitura:** espera-se uma troca: mais peso reduz o vazamento, mas aumenta o erro dentro da tarefa. Se o resultado for estável entre os três valores, o método não depende de um ajuste fino desse parâmetro.

**A3. Posto do banco de gaussianas.**
- **Pergunta:** quanto detalhe da covariância de cada classe é preciso guardar?
- **Como:** braço gauss com posto da covariância em {8; 32}. O valor principal é 32. Posto 8 guarda cerca de um quarto da memória.
- **Leitura:** se o posto 8 der resultado parecido, o método pode guardar bem menos memória por classe.

**A4. Como escolher os pseudo-negativos.**
- **Pergunta:** vale a pena escolher os pseudo-negativos que o adapter novo está "roubando", em vez de sorteá-los ao acaso?
- **A margem de vazamento.** Para uma feature sintética z da classe antiga c, a margem é

  m(z, c) = (maior nota que alguma classe **nova** dá a z) − (nota que a classe **verdadeira** c dá a z).

  Se m > 0, alguma classe nova vence a classe verdadeira: z **vazaria**. Quanto maior m, mais o adapter novo está "roubando" z. Se m < 0, a classe verdadeira vence, e o adapter novo já recusa z.
- **Como a escolha é feita, a cada lote de treino:**
    1. Sorteiam-se M = 4096 **candidatos** do banco de gaussianas (classes antigas escolhidas ao acaso, e uma feature sorteada da gaussiana de cada uma).
    2. Para cada candidato i, calcula-se a margem m_i com o adapter novo **no estado atual** do treino.
    3. Cada candidato recebe o peso w_i = exp(β × m_i), e os pesos são normalizados para somar 1: p_i = w_i / (w_1 + … + w_M).
    4. Sorteiam-se K = 48 candidatos, com reposição, cada um com probabilidade p_i. Esses 48 são os pseudo-negativos do lote.

  Esse procedimento é chamado de amostragem por importância com reamostragem: os candidatos vêm de uma distribuição fácil de sortear (o banco), e os pesos corrigem a escolha na direção da distribuição que queremos (a dos que vazam).
- **O papel de β.** β controla **o quanto** a escolha favorece os candidatos que vazam:
    - **β = 0:** todos os pesos valem exp(0) = 1, todos os candidatos têm a mesma chance, e o resultado é igual ao sorteio direto. É o braço gauss.
    - **β grande:** os pesos crescem muito depressa com a margem, e quase só os candidatos que mais vazam são escolhidos.
    - **Exemplo:** dois candidatos, um com margem +0,2 (vaza) e outro com −0,2 (não vaza). A razão entre os pesos é exp(β × 0,4). Com β = 5, ela vale exp(2) ≈ 7,4: o que vaza tem 7,4 vezes mais chance de ser escolhido. Com β = 0,5, vale exp(0,2) ≈ 1,2: quase a mesma chance.
    - **Por que β = 0,5 em margem_padronizada e β = 5 em margem:** as notas brutas são cossenos, e as margens entre elas são da ordem de décimos; as notas padronizadas são medidas em desvios-padrão, e as margens entre elas são da ordem de unidades, cerca de dez vezes maiores. Usar β dez vezes menor nas notas padronizadas mantém um grau de preferência da mesma ordem nos dois braços.
    - **Cuidado com β alto demais:** se poucos candidatos concentram quase todo o peso, os 48 escolhidos são quase sempre os mesmos, e o treino vê pouca variedade. Medimos isso pelo **tamanho efetivo da amostra**, (Σ w_i)² / Σ w_i², dividido por M. No desenvolvimento, ele ficou com mediana de 17% e mínimo de 8%, longe desse problema.
- **As três formas comparadas, com o resto igual:**
    - **gauss** (β = 0): sorteio direto do banco;
    - **margem** (β = 5): margem calculada com notas brutas;
    - **margem_padronizada** (β = 0,5): margem calculada com notas padronizadas, as mesmas usadas para decidir a classe no teste. Como a tarefa em treino ainda não terminou, μ e σ dela são estimados durante o treino por uma média móvel sobre os lotes.
- **Leitura:** responde à questão secundária S2. No desenvolvimento não houve diferença significativa.

**Comparação embutida, sem execução extra:** cenários 3 e 4 mostram o efeito do número de tarefas (5 contra 20) com o mesmo backbone e dataset.

### 4.5 Análise exploratória condicionada: o diagnóstico aplicado aos métodos publicados

**Condição:** só é feita se EASE, MOS e TUNA forem rodados por nós (Seção 4.3.7). Com os números publicados, não há como fazê-la, porque ela precisa da predição de cada imagem. Por ser exploratória, não tem previsão e não entra em P1 a P4; ela é descrita aqui para que, se for feita, não conte como análise inventada depois de ver os dados.

**Motivação.** Nossos adapters ficam depois do backbone, e não dentro dos blocos do ViT como nos três métodos, porque o mecanismo de recusa precisa de um espaço de features fixo onde as gaussianas das classes antigas continuem válidas (Seção 4.3.7 e artigo, Seção de Método). Um revisor pode perguntar se o diagnóstico vale para esses métodos reais ou só para o nosso sistema simples. Esta análise responde a essa pergunta.

**Como cada método escolhe a tarefa no teste** (conferido no código do LAMDA-PILOT em 23/11/2026):
- **EASE:** junta as features de todos os adapters num único vetor e classifica com protótipos sobre esse vetor. Não há uma escolha explícita de adapter.
- **MOS:** faz uma primeira predição com o modelo de adapters fundidos, vai para o adapter da tarefa da classe predita, prediz de novo e repete até a tarefa não mudar (no máximo 4 vezes). No fim, soma essa predição à do modelo fundido.
- **TUNA:** cada adapter de tarefa produz uma predição; escolhe-se o adapter cuja predição tem a **menor entropia** (a mais "confiante"), e soma-se a predição dele à do adapter universal.

**O que medimos, nos cenários 4 a 8, nas seeds em que os métodos forem rodados:**

1. **Diagnóstico de vazamento, nos três métodos:** vazamento Λ, erro dentro da tarefa E_WP, matriz de vazamento e índice de direção D, calculados exatamente como na Seção 7.2, a partir da classe predita e da classe verdadeira de cada imagem de teste. Pergunta: o vazamento desses métodos também é maior que o do NCM e também vai para as tarefas mais novas?
2. **Acerto da escolha de adapter, no MOS e no TUNA:** fração das imagens de teste em que o adapter escolhido é o da tarefa verdadeira. É a medida mais direta de vazamento no momento da escolha.
3. **Padronização da escolha, só no TUNA:** a escolha pela menor entropia supõe que as entropias dos adapters estão na mesma escala, a mesma suposição que a nossa padronização corrige. Para cada adapter k, calculamos a média μ_k e o desvio-padrão σ_k da entropia das suas predições nas imagens de treino da tarefa k, ao fim da tarefa k. No teste, escolhemos o adapter de menor entropia padronizada, (entropia − μ_k)/σ_k. Comparamos a acurácia final e o acerto da escolha com e sem essa padronização, no **mesmo** modelo treinado. Também calculamos a dispersão de escala DE das entropias (Seção 7.2, com as entropias no lugar das notas).
4. **No MOS e no EASE, a padronização não é aplicada:** o MOS escolhe o adapter pela classe predita, e não comparando notas entre adapters, e o EASE não escolhe adapter. Não há uma comparação de escalas para corrigir sem mudar o método.

**Mudança no código dos métodos:** apenas registro. Acrescentamos ao código do PILOT a gravação, por imagem de teste, da classe verdadeira, da classe predita, do adapter escolhido (MOS e TUNA) e das entropias por adapter (TUNA). Nenhuma linha que afeta treino ou predição é alterada, e o código acrescentado é publicado junto com o nosso. A padronização do item 3 é calculada depois, a partir desses registros.

**Estatística:** as mesmas da Seção 7 (diferenças pareadas por seed, teste t, intervalo de 95%), com a ressalva de que tudo aqui é exploratório e é rotulado assim no artigo.

---

## 5. Hiperparâmetros congelados

### 5.1 Comuns a todos os cenários (valores das execuções de desenvolvimento)

| Parâmetro | Valor | O que é |
|---|---|---|
| Adapter | residual, d → 64 → d, projeção de subida iniciada em 0 | O adapter soma uma correção a φ; começa como a identidade |
| Classificador | cosseno a protótipos; protótipos novos iniciados na média de φ e recalculados a cada época | — |
| Escala dos logits s | 16 | Multiplica os cossenos dentro da perda |
| Otimizador | SGD, taxa 0,01, momentum 0,9, weight decay 0 | — |
| Lote | 48 imagens | — |
| Épocas por tarefa | 100, sem parada antecipada | — |
| Aumento de dados | nenhum | Por isso φ pode ficar em cache |
| Pseudo-negativos por lote, K | 48 | — |
| Peso da recusa, λ_pn | 0,5 | Peso da segunda parcela da perda |
| Banco de gaussianas | covariância de posto 32 mais um termo isotrópico; encolhimento 10⁻³ | Guarda as 32 direções principais de variação de cada classe |
| Margem / margem_padronizada | 4096 candidatos; β = 5 / β = 0,5 | — |
| Padronização | μ_k, σ_k da maior nota do adapter k em cada imagem de treino da tarefa k, calculados ao fim da tarefa k | — |

### 5.2 Mudanças obrigadas pela arquitetura

Decididas agora, antes de qualquer execução com ViT.

| Parâmetro | ResNet-18 | ViT-B/16 | Motivo |
|---|---|---|---|
| Checkpoint | torchvision, ImageNet-1k | `vit_base_patch16_224_in21k` (timm), ImageNet-21k | O mesmo de SimpleCIL e EASE, para comparar com a literatura |
| d | 512 | 768 | Tamanho da feature de cada rede |
| Transformação da imagem | CIFAR: Resize(64). Pastas: Resize(73) + CenterCrop(64). Normalização ImageNet | Resize(256, bicúbica) + CenterCrop(224); sem normalização, como no PILOT | Comparabilidade |
| ReLU nas amostras do banco | sim | **não** | A feature da ResNet nunca é negativa (sai de uma ReLU), então amostras negativas seriam irreais. A do ViT pode ser negativa |

Nenhum outro hiperparâmetro muda entre backbones ou datasets. Se o treino falhar com o ViT (perda que não diminui nas primeiras épocas da primeira tarefa), a correção é escolhida usando só os 10% de validação de uma única seed (1997) e documentada em adendo antes de qualquer outra execução.

---

## 6. Métricas registradas

Em cada execução, depois de cada tarefa e no fim: acurácia, vazamento Λ, erro dentro da tarefa E_WP, matriz de vazamento L, índice de direção D, dispersão de escala DE, parcela de escala κ, acurácia com oráculo de tarefa e esquecimento médio. Também: memória guardada por classe, tempo de treino por tarefa e os valores μ_k e σ_k de cada tarefa (necessários para a dispersão de escala DE; não eram gravados nas execuções de desenvolvimento). O cálculo de cada uma está na Seção 7.2.

---

## 7. Análise

### 7.1 Ideia geral

Cada seed é uma **repetição** do experimento. Dentro de uma seed, todos os braços veem exatamente a mesma ordem de classes. Por isso as comparações são feitas **seed a seed** (comparação **pareada**): primeiro calculamos a diferença entre dois braços em cada seed; depois perguntamos se essas seis diferenças são consistentemente diferentes de zero.

Exemplo: se na seed 1997 o braço A acerta 44% e o B 42%, a diferença é +2 pp. Fazemos isso nas seis seeds e analisamos as seis diferenças. Isso elimina a variação que vem da ordem das classes (algumas ordens são mais difíceis para todos os braços), o que torna a comparação muito mais sensível.

### 7.2 Como cada métrica é calculada

Todas são calculadas sobre as imagens de teste de todas as classes vistas até o momento.

- **Acurácia:** número de imagens classificadas corretamente dividido pelo total.
- **Acurácia final:** a acurácia depois da última tarefa.
- **AAC:** a média das acurácias medidas depois de cada tarefa. Com 5 tarefas, é a média de 5 números.
- **Vazamento Λ:** número de imagens cuja classe predita pertence a outra tarefa, dividido pelo total.
- **Erro dentro da tarefa E_WP:** número de imagens cuja classe predita é errada, mas da tarefa certa, dividido pelo total. Vale sempre: 1 − acurácia = Λ + E_WP.
- **Matriz de vazamento L:** tabela T × T em que a linha i, coluna j, é a fração das imagens da tarefa i classificadas numa classe da tarefa j. A diagonal são as imagens que ficaram na tarefa certa.
- **Índice de direção D:** a soma das entradas de L acima da diagonal (imagens que foram para tarefas mais novas) dividida pela soma das entradas abaixo da diagonal (imagens que foram para tarefas mais antigas).
- **Dispersão de escala DE:** para cada tarefa k, temos μ_k e σ_k (média e desvio-padrão da maior nota do adapter k nas suas imagens de treino). DE = desvio-padrão dos μ_k entre as tarefas ÷ média dos σ_k. Exemplo: se as médias das tarefas forem 0,60, 0,70 e 0,80 (desvio-padrão 0,10) e os desvios típicos forem 0,05, DE = 0,10 / 0,05 = 2: as médias diferem em duas vezes o espalhamento típico de cada adapter.
- **Parcela de escala κ:** (Λ − Λ̃) / Λ, em que Λ̃ é o vazamento com notas padronizadas.
- **Acurácia com oráculo de tarefa:** acurácia quando a decisão é restrita às classes da tarefa verdadeira de cada imagem. Mede só a qualidade dentro da tarefa.
- **Esquecimento médio:** para cada tarefa (menos a última), a maior acurácia que ela teve em algum momento menos a acurácia dela no fim; depois, a média entre as tarefas.
- **Memória por classe:** número de valores guardados por classe × 4 bytes (números de 32 bits). Exemplo: o banco de posto 32 com d = 768 guarda (32 + 1) × 768 + 33 = 25.377 valores ≈ 101 KB por classe.
- **Tempo por tarefa:** tempo de relógio do treino de cada tarefa, medido pelo próprio script.

### 7.3 Teste t pareado, passo a passo

Usado em P1, P3, P4 e S1a (e em S1b, se os métodos publicados forem rodados por nós). Sejam A e B dois braços e n = 6 seeds.

1. **Diferença em cada seed:** d_s = resultado de A − resultado de B, para s = 1, …, 6.
2. **Média das diferenças:** d̄ = (d_1 + … + d_6) / 6. É o efeito estimado, em pp.
3. **Desvio-padrão das diferenças:** s_d = raiz de [ Σ (d_s − d̄)² / (n − 1) ]. Mede o quanto as diferenças variam de uma seed para outra.
4. **Erro-padrão da média:** EP = s_d / √n. Mede a incerteza sobre d̄.
5. **Estatística t:** t = d̄ / EP. Diz quantos "erros-padrão" a média está longe de zero.
6. **Valor p:** a probabilidade de obter um t tão longe de zero quanto o observado **se** os dois braços fossem, na verdade, equivalentes. Calculado pela distribuição t de Student com n − 1 = 5 graus de liberdade, bicaudal (conta afastamentos para os dois lados). Valor p pequeno (abaixo de 0,05, depois da correção da Seção 7.5) = diferença considerada real.
7. **Intervalo de confiança de 95% (IC 95%):** IC = d̄ ± 2,571 × EP.
    - **O que ele expressa.** d̄ é a nossa melhor estimativa do efeito, mas com outras seis seeds sairia um valor um pouco diferente. O intervalo mostra a faixa de valores do efeito **verdadeiro** que são compatíveis com os dados. Em termos técnicos: se o experimento fosse repetido muitas vezes, com seis seeds novas a cada vez, e calculássemos o intervalo da mesma forma, 95% desses intervalos conteriam o efeito verdadeiro.
    - **De onde vem o 2,571.** Se tivéssemos muitas seeds, usaríamos 1,96, o valor da distribuição normal que deixa 2,5% de cada lado. Com só seis seeds, o próprio desvio-padrão s_d é estimado com incerteza, e a distribuição correta é a t de Student com n − 1 = 5 graus de liberdade, que tem "caudas" mais largas. O valor que deixa 2,5% em cada cauda dessa distribuição é 2,571. Por isso o intervalo com seis seeds é cerca de 30% mais largo do que seria com muitas seeds.
    - **Como calcular, no exemplo do efeito claro abaixo:** d̄ = 14,85 e EP = 0,11. Então 2,571 × 0,11 = 0,28, e IC = 14,85 ± 0,28 = [14,57; 15,12].
    - **Como ler a largura.** Intervalo estreito = efeito medido com precisão. Intervalo largo = muita variação entre seeds, efeito impreciso.
    - **Relação com o valor p.** Se o intervalo **não contém zero**, o valor p (antes da correção) é menor que 0,05; se contém zero, é maior. As duas informações são coerentes; o intervalo diz, além disso, o tamanho plausível do efeito.
    - **Os intervalos reportados não são corrigidos por Holm.** A decisão de "significativo ou não" usa os valores p corrigidos (Seção 7.5); os intervalos servem para mostrar tamanho e precisão do efeito.

**Exemplo com efeito claro** (desenvolvimento: baseline padronizado − baseline bruto, seeds 1993–1998):
diferenças = 14,54; 15,25; 15,06; 14,65; 14,74; 14,83 pp → d̄ = 14,85; s_d = 0,27; EP = 0,11; t = 137; p < 0,001; IC 95% = [14,57; 15,12]. Todas as seeds concordam, e o efeito é enorme em relação à variação.

**Exemplo com efeito incerto** (margem_padronizada − gauss, ambos padronizados):
diferenças = 2,13; 0,55; 0,17; 0,21; 1,43; −0,23 pp → d̄ = 0,71; s_d = 0,89; EP = 0,36; t = 1,95; p = 0,11; IC 95% = [−0,23; 1,65]. A média é positiva, mas o intervalo contém zero: não dá para afirmar que há efeito.

**Por que o teste t e não o de Wilcoxon:** o teste de Wilcoxon, que não supõe distribuição normal, com 6 seeds não consegue dar valor p menor que 0,031. Depois de qualquer correção para várias comparações, nenhum resultado poderia ser significativo. O número de seeds com o sinal esperado (Seção 7.7) cumpre o papel de verificação sem suposições.

### 7.4 Aplicação às previsões

| Previsão | A | B | Métrica | Onde |
|---|---|---|---|---|
| P1(a) | baseline (bruto) | ncm | vazamento Λ | cada cenário |
| P3(a) | gauss (padronizado) | baseline (padronizado) | acurácia final | cada cenário |
| P3(b) | gauss (padronizado) | gauss_ca | acurácia final | cada cenário |
| P4(a) | adapter_compartilhado | baseline | Δ_pad | cenários 1, 2, 3, 5 |
| S1a | gauss (padronizado) | ncm e fecam (cada um) | acurácia final | cada cenário |
| S1b (só se rodados por nós) | gauss (padronizado) | ranpac, ease, mos, tuna (cada um) | acurácia final | cenários 4 a 8 |

### 7.5 Correção para várias comparações (Holm)

**O problema:** com o limite de 5%, cada teste tem 5% de chance de dar "significativo" por puro acaso, mesmo sem efeito real. Com 8 cenários, a chance de pelo menos um falso positivo passa de 33%. A correção de Holm controla isso: garante que a chance de **qualquer** falso positivo dentro de uma família de testes fique em no máximo 5%.

**Famílias:** cada previsão é uma família. P1(a), P3(a) e P3(b) têm 8 testes cada (um por cenário); P4(a) tem 4; P2 tem 2 (um por preditor); S1a tem um teste por cenário e por método comparado.

**Procedimento, para uma família com m testes:**
1. Ordenar os m valores p do menor para o maior: p(1) ≤ p(2) ≤ … ≤ p(m).
2. Comparar p(1) com 0,05/m. Se for menor, o teste 1 é significativo; passar ao próximo. Se não, parar: nenhum teste é significativo.
3. Comparar p(2) com 0,05/(m − 1). Se for menor, o teste 2 é significativo; passar ao próximo. Se não, parar.
4. Continuar assim, com limite 0,05/(m − k + 1) para o k-ésimo teste, até o primeiro que falhar. Ele e todos os seguintes não são significativos.

**Exemplo com 3 cenários**, valores p 0,010; 0,030; 0,045:
- 0,010 < 0,05/3 = 0,0167 → significativo;
- 0,030 < 0,05/2 = 0,025? Não → parar.
Só o primeiro cenário é significativo, embora os três tenham p < 0,05 isoladamente.

O artigo reporta o valor p ajustado, que equivale ao procedimento acima: o valor p ajustado do k-ésimo teste é o maior entre (m − k + 1) × p(k) e o ajustado do teste anterior, limitado a 1.

### 7.6 Correlação de Spearman e teste de permutação (P2)

**O que é:** a correlação de Spearman mede se duas grandezas crescem juntas, sem exigir uma relação em linha reta. Ela vai de −1 (uma sobe enquanto a outra desce) a +1 (sobem juntas na mesma ordem).

**Como calcular, passo a passo:**
1. Para cada um dos 8 cenários: Δ_pad médio nas seis seeds e o preditor (mediana de D, ou média de DE).
2. Substituir os valores por suas **posições**: o menor Δ_pad recebe posição 1, o maior posição 8; o mesmo para o preditor. Empates recebem a média das posições.
3. Calcular a correlação comum (Pearson) entre as duas listas de posições. O resultado é ρ (rho) de Spearman.

**Teste de permutação unilateral:** pergunta-se com que frequência uma correlação tão alta quanto a observada apareceria se não houvesse relação nenhuma. Para isso, embaralha-se a associação entre os 8 valores de Δ_pad e os 8 valores do preditor de **todas** as formas possíveis (8! = 40.320 embaralhamentos), calcula-se ρ em cada uma, e o valor p é a fração dos embaralhamentos com ρ maior ou igual ao observado. "Unilateral" porque a previsão tem direção (correlação positiva). Com 8 pontos, o cálculo é exato, sem aproximações.

**Intervalo de confiança por bootstrap:** para mostrar a incerteza de ρ, repete-se 10.000 vezes o seguinte: em cada cenário, sortear 6 seeds com reposição entre as 6 disponíveis (algumas aparecem duas vezes, outras nenhuma), recalcular as médias de cada cenário e o ρ. O intervalo de 95% vai do percentil 2,5 ao percentil 97,5 dos 10.000 valores de ρ.

**Correção:** Holm com 2 testes (D e DE).

### 7.7 Medidas complementares, sempre reportadas

- **Seeds com o sinal esperado:** quantas das seis seeds mostram a diferença na direção prevista (por exemplo, 6/6). É uma verificação simples, que não depende de suposições estatísticas.
- **Tamanho do efeito:** a média das diferenças em pp, com o intervalo de 95%. Um efeito pode ser significativo e pequeno demais para importar, ou grande e incerto; os dois números juntos mostram isso.

### 7.8 Regras gerais

- **Tudo é reportado:** previsões confirmadas, refutadas e inconclusivas, com o mesmo destaque.
- **Análises não listadas aqui** aparecem no artigo rotuladas como exploratórias. As ablações (Seção 4.4) e a análise da Seção 4.5 já são, por definição, exploratórias.
- **Execuções com erro** são repetidas com a mesma seed. Nenhuma execução concluída é descartada.

### 7.9 Seeds 1997 e 1998 no cenário 1

Como já foram observadas, as conclusões do cenário 1 são reportadas duas vezes: com as seis seeds de teste e só com as quatro nunca vistas (1999–2002). Se as duas análises discordarem, vale a de quatro seeds.

### 7.10 Resumo das regras de decisão

| Previsão | Confirmada quando | Refutada quando |
|---|---|---|
| P1 | em cada cenário: excesso de vazamento > 0 e significativo (Holm, 8 cenários) e mediana de D > 1; global: todos os cenários | excesso ≤ 0 ou D ≤ 1 |
| P2 | ρ > 0 com p < 0,05 (permutação, Holm com 2 testes) para D ou DE | ρ ≤ 0 para ambos |
| P3 | em cada cenário: gauss > baseline e gauss > gauss_ca, ambos significativos (Holm, 8 cenários) | alguma das diferenças ≤ 0 |
| P4 | nos 4 cenários: Δ_pad compartilhado < Δ_pad baseline, significativo (Holm, 4), e |Δ_pad compartilhado| < 1 pp | Δ_pad compartilhado ≥ Δ_pad baseline em algum cenário |

---

## 8. Verificações de protocolo

### 8.1 O NCM reproduz o SimpleCIL publicado (antes de olhar qualquer resultado das previsões)

A acurácia final do NCM não depende do tamanho das tarefas: depois da última tarefa, ele decide entre todas as classes do mesmo jeito. Por isso, com o ViT-B/16-IN21K, ela deve reproduzir o SimpleCIL publicado:

| Dataset | SimpleCIL publicado | Fonte |
|---|---|---|
| CIFAR-100 | 81,26% | EASE (Tabela 1); TUNA (Tabela 1) |
| ImageNet-R | 54,55% | EASE (Tabela 1); TUNA (Tabela 1) |
| ImageNet-A | 48,91% | EASE (Tabela 1); TUNA (Tabela 1) |
| OmniBenchmark | 73,15% | EASE (Tabela 1) |
| VTAB | 84,38% | EASE (Tabela 1) |

Uma diferença maior que 2 pp em algum dataset indica erro de protocolo (dados, transformação ou checkpoint). Nesse caso, o erro é corrigido e documentado em adendo antes de qualquer análise das previsões, e as execuções afetadas são refeitas. Se a correção mudar o código, ele recebe uma nova tag.

### 8.2 Reprodução de um método publicado

Para mostrar que os números copiados são comparáveis aos nossos, rodamos o **RanPAC** no ImageNet-R (cenário 5), com o código e a configuração do LAMDA-PILOT e a seed 1993. É o mais barato dos quatro métodos e um dos mais fortes. O valor publicado é 77,94% de acurácia final (TUNA, Tabela 1). Se a nossa execução ficar a até 2 pp disso, consideramos a cópia de números validada. Se não ficar, investigamos a diferença, e o artigo passa a reportar os números publicados com essa ressalva explícita.

---

## 9. Código

**9.1 Plano e código das execuções de teste.** Este plano (versão 1.0) e o código das execuções de teste ficam no repositório `eavcampos/CIL_e_GFlowNet`. Antes da primeira execução de teste, o código é marcado com a tag git `v1.0-teste`, e a tag é enviada ao GitHub. O commit dessa tag contém este plano sem alterações. Um adendo (Seção 10) informa o hash do commit. As execuções de teste só usam esse código.

**9.2 O que pode e o que não pode ser conferido.** Depois da publicação, o repositório completo fica público: qualquer leitor pode ver o histórico, as tags `v0.9-desenvolvimento` e `v1.0-teste`, e comparar o plano, o código e os resultados. Durante uma revisão duplo-cega, o avaliador recebe o código por um espelho anônimo (por exemplo, Anonymous GitHub), que não mostra o histórico, e este plano como apêndice do material suplementar. Em nenhum dos dois casos as datas podem ser confirmadas por uma fonte independente dos autores, porque não houve depósito externo (Seção 0). O artigo diz isso explicitamente e descreve o procedimento como "plano de análise definido antes das execuções de teste e versionado no repositório", sem afirmar uma verificação independente.

**9.3 Versão limpa para o artigo.** Perto da submissão, é criada uma versão do repositório só com o que o artigo usa, marcada com a tag `v1.0-artigo`. Ela só pode remover arquivos não usados, renomear e documentar; nenhuma linha que afete treino ou avaliação pode mudar. A equivalência é verificada rodando uma seed com `v1.0-artigo` e conferindo que os resultados são idênticos aos da mesma seed com `v1.0-teste`. O espelho anônimo da revisão e o DOI de arquivamento (Zenodo), depois da aceitação, são gerados a partir de `v1.0-artigo`.

---

## 10. Adendos

Qualquer desvio deste plano é descrito aqui, com data, motivo e efeito esperado, num commit próprio feito **antes** de o desvio ser aplicado.

| Data | Desvio | Motivo |
|---|---|---|
| — | — | — |

---

## Anexo A — Mapa de fontes dos números publicados

Acurácia final (%), ViT-B/16-IN21K, ordem de classes da seed 1993, sem exemplares. "TUNA T1" = Tabela 1 do artigo do TUNA; "EASE T1" e "MOS T1", idem. Toda célula deve ser conferida no PDF original antes do uso no artigo.

| Método | CIFAR-100 B0 Inc5 | ImageNet-R B0 Inc20 | ImageNet-A B0 Inc20 | OmniBenchmark B0 Inc30 | VTAB B0 Inc10 |
|---|---|---|---|---|---|
| SimpleCIL | 81,26 (TUNA T1) | 54,55 (TUNA T1) | 48,91 (TUNA T1) | 73,15 (EASE T1) | 84,38 (EASE T1) |
| RanPAC | 90,62 (TUNA T1) | 77,94 (TUNA T1) | 61,82 (TUNA T1) | — (sem fonte com este protocolo; o artigo do RanPAC usa outros) | — (idem) |
| EASE | 85,80 (TUNA T1) | 76,17 (TUNA T1) | 55,04 (TUNA T1) | 74,85 (EASE T1) | 93,55 (EASE T1) |
| MOS | 89,25 (MOS T1) | 77,93 (MOS T1) | 59,12 (MOS T1)† | 80,05 (MOS T1) | 92,79 (MOS T1) |
| TUNA | 90,74 (TUNA T1) | 79,42 (TUNA T1) | 64,78 (TUNA T1) | — (sem fonte com este protocolo) | — (sem fonte com este protocolo) |

Observações:
- O artigo do EASE reporta ImageNet-R com Inc5; por isso o número do EASE no ImageNet-R vem da tabela do TUNA, que usa Inc20.
- A AAC publicada de cada método é extraída das mesmas tabelas.
- **Regra:** quando o próprio artigo do método reporta o número com o mesmo protocolo, usa-se o dele; senão, o da tabela mais recente com o mesmo protocolo. Discrepâncias entre fontes são anotadas aqui e no artigo.
- Valores do MOS conferidos em 23/11/2026 na Tabela 1 do artigo do MOS (versão HTML do arXiv), cujas colunas são, para cada dataset, Ā e depois 𝒜_B (acurácia final).
- † **Discrepância:** o artigo do MOS reporta 59,12% no ImageNet-A; a tabela do TUNA reporta 56,22% para o MOS no mesmo protocolo. Usamos o número do próprio MOS e mencionamos a diferença.
- O ObjectNet aparece nas tabelas do MOS (Inc10) e do TUNA (Inc20) com os mesmos valores para EASE e MOS, o que sugere que um dos rótulos está errado; ele foi excluído por outro motivo (licença).
- O RanPAC não aparece na tabela principal do MOS.

---

## Referências citadas

- ObjectNet. Licença de uso, distribuída com o dataset (cópia em huggingface.co/datasets/timm/objectnet, arquivo LICENSE), consultada em 22/11/2026.
- Goswami, D. et al. FeCAM: Exploiting the Heterogeneity of Class Distributions in Exemplar-Free Continual Learning. NeurIPS, 2023.
- Sun, H.-L.; Zhou, D.-W.; Zhan, D.-C.; Ye, H.-J. PILOT: A Pre-Trained Model-Based Continual Learning Toolbox. SCIENCE CHINA Information Sciences, 68(4):147101, 2025.
- McDonnell, M. D. et al. RanPAC: Random Projections and Pre-trained Models for Continual Learning. NeurIPS, 2023.
- Sun, H.-L. et al. MOS: Model Surgery for Pre-Trained Model-Based Class-Incremental Learning. AAAI, 2025.
- Wang, Y. et al. Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning (TUNA). ICCV, 2025.
- Xu, L.; Zhou, Z.; Ji, W.; Miao, C.; Zhao, P.; Zhang, L. Distribution-Conditioned Task Routing for Class-Incremental Learning. arXiv 2609.34503v1, submetido em 28/09/2026 (conferido em 23/11/2026). A padronização é testada na Seção 3.2 e na Tabela 1 deles: com um encoder ViT-B/16 compartilhado e incrementos LoRA cumulativos, a acurácia muda +0,35; −0,02; −0,79; −0,54 e +0,11 pp em CIFAR-100, ImageNet-A, ImageNet-R, CUB-200 e OmniBenchmark.
- Zhang, G. et al. SLCA: Slow Learner with Classifier Alignment for Continual Learning on a Pre-trained Model. ICCV, 2023.
- Zhou, D.-W. et al. Expandable Subspace Ensemble for Pre-Trained Model-Based Class-Incremental Learning (EASE). CVPR, 2024.
- Zhou, D.-W. et al. Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need (SimpleCIL/APER). IJCV, 2024.
