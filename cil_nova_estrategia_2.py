### nova_estrategia_2 - Nova Estratégia 1 + PSEUDO-NEGATIVOS no espaço phi do backbone congelado
"""
NOVA ESTRATÉGIA 2: NOVA ESTRATÉGIA 1 + PSEUDO-NEGATIVOS
================================================================================

Idêntica à Nova Estratégia 1 (abaixo), com três acréscimos, todos marcados no código com
"[Estratégia 2]":

  1. BANCO DE GAUSSIANAS: ao fim de cada tarefa, guarda média e covariância (posto baixo +
     diagonal) de phi = backbone(imagem) para cada classe da tarefa. phi vem da ResNet-18
     CONGELADA, então essas estatísticas nunca envelhecem. Nenhuma imagem é guardada.

  2. PSEUDO-NEGATIVOS NA CE: na tarefa t >= 1, cada lote real recebe também phi SINTÉTICOS das
     classes antigas, com o rótulo antigo, avaliados exatamente como os reais (Adapters antigos
     para as classes antigas, Adapter_t para as novas, score cosseno + LVQ). Como os Adapters e
     as unidades LVQ antigas estão congelados, o gradiente só pode BAIXAR o score das classes
     novas para esses phi: é o "isto não é meu" que hoje o Adapter_t nunca aprende.
       perda = (1 - lambda_lvq) * CE_real + lambda_lvq * L_LVQ + lambda_pn * CE_pseudo
     Modos (config_estrategia_2.json -> pseudo_negativos.modo):
       "nenhum" -> reproduz a Nova Estratégia 1 (linha de base);
       "gauss"  -> sorteio direto das gaussianas;
       "is"     -> amostragem por importância focada onde há vazamento (margem > 0);
       "gfn"    -> GFlowNet híbrida (classe discreta -> phi contínuo) com buffer do "is".

  3. MÉTRICAS DE VAZAMENTO (metricas_vazamento.py): taxa de vazamento Lambda, decomposição
     exata 1 - Acc = Lambda + E_WP, teto de ganho com oráculo de tarefa, matriz L, índice de
     direção (recência), correlação semântica e parcela atribuível à escala (kappa), após
     cada fronteira e no final. Complementam a "margem de vazamento" já existente.
"""
"""
NOVA ESTRATÉGIA 1: GENERALIZAÇÃO DA ESTRATÉGIA 26
================================================================================

Mantém, sem alteração de princípio, a arquitetura já validada nas Estratégias
24e-27c deste projeto: um backbone ResNet-18 pré-treinado no ImageNet,
congelado desde a criação, e um módulo Adapter (bottleneck) NOVO para cada
tarefa, treinado uma única vez e depois CONGELADO e MANTIDO PERMANENTEMENTE —
nenhum Adapter de uma tarefa já concluída volta a receber gradiente quando uma
tarefa nova começa. A classificação final combina, por forma convexa,
similaridade de cosseno contra um protótipo de classe com um score de um mapa
LVQ hexagonal supervisionado (Sato & Yamada, 1995), exatamente como na
Estratégia 25c/25c_v2/26. O diagnóstico A/B/C de Task Arithmetic (Ilharco et
al., ICLR 2023) e o cálculo de drift semântico exato (Yu et al., 2020, Eq. 8)
também são mantidos, generalizados para qualquer número de fronteiras
tarefa->tarefa (a Estratégia 26 só cobria duas fronteiras: M1->M2 e M2->M3).

O que esta estratégia GENERALIZA em relação à Estratégia 26 (que tinha tudo
fixo: 3 tarefas, 2 classes por tarefa, sempre CIFAR-10 cat/dog/deer/
automobile/frog/ship, batch_size=64, otimizador Adam fixo):

  1. DATASET plugável — 'cifar10', 'cifar100' ou 'omnibenchmark' (este último
     em formato ImageFolder, train/ e test/, uma subpasta por classe —
     mesmo benchmark de 300 classes fine-grained citado em SimpleCIL/ADAM,
     Zhou et al. IJCV 2024, já referenciado nas Estratégias 24c-24e).
  2. NÚMERO DE TAREFAS e CLASSES POR TAREFA configuráveis (config.json:
     tarefas.num_tasks, tarefas.classes_por_tarefa) — generaliza as 3
     tarefas fixas de 2 classes da Estratégia 26 para qualquer N.
  3. SELEÇÃO DE CLASSES por sorteio determinístico: dado o dataset e
     num_tasks*classes_por_tarefa, as classes que compõem cada tarefa são
     escolhidas aleatoriamente (numpy.random.RandomState) a partir da seed
     definida em config.json ("seed"). O estudo planejado usa 4 seeds
     (1993, 1994, 1995, 1996, documentadas em "seeds_experimento_planejados"
     no config), uma por execução completa do script.
  4. BATCH SIZE configurável, default 48 (config.json: treino.batch_size).
  5. OTIMIZADOR configurável — SGD por default (config.json: otimizador.*),
     com suporte a Adam/AdamW/RMSprop.

A grade do mapa LVQ hexagonal, antes fixa em 12x8=96 unidades (3 blocos de
linha x 2 blocos de coluna, 16 unidades/classe — ver docstring da Estratégia
26), agora é dimensionada automaticamente a partir de num_tasks,
classes_por_tarefa e unidades_lvq_por_classe (reduz-se exatamente à grade da
Estratégia 26 quando num_tasks=3, classes_por_tarefa=2,
unidades_lvq_por_classe=16).

Esta execução salva checkpoint (todos os Adapters, FC, LVQ completo, thetas
de inicialização de cada Adapter/LVQ-por-tarefa, protótipos congelados e a
partição de classes sorteada) em resultados/checkpoints/, para permitir reavaliação
sem retreinar.
"""
import os
import sys
import json
import math
import copy
import random
import logging
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# [Estratégia 2]
from pseudo_negativos_cil import (BancoGaussianas, GeradorPseudoNegativos, coletar_phi_por_classe,
                                  scores_combinados)
from metricas_vazamento import leakage_metrics

# ============================================================================
# 0. LINHA DE COMANDO + CONFIGURAÇÃO
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def carregar_configuracao():
    parser = argparse.ArgumentParser(description="Nova Estratégia 2 (generalização da Estratégia 26)")
    parser.add_argument("--config", default=os.path.join(SCRIPT_DIR, "config_estrategia_2.json"),
                         help="Caminho para o config.json (default: config.json ao lado deste script)")
    parser.add_argument("--seed", type=int, default=None,
                         help="Sobrescreve config['seed'] (conveniência para rodar as 4 seeds planejadas "
                              "sem editar o JSON a cada execução)")
    args, _ = parser.parse_known_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if args.seed is not None:
        cfg["seed"] = args.seed
    cfg["_arquivo_config"] = os.path.abspath(args.config)   # [Estratégia 2] nomeia as saídas
    return cfg


CFG = carregar_configuracao()
CFG_DATASET = CFG["dataset"]
CFG_TAREFAS = CFG["tarefas"]
CFG_TREINO = CFG["treino"]
CFG_OTIMIZADOR = CFG["otimizador"]

SEED = int(CFG["seed"])
NUM_TASKS = int(CFG_TAREFAS["num_tasks"])
CLASSES_POR_TAREFA = int(CFG_TAREFAS["classes_por_tarefa"])
NUM_CLASSES_TOTAIS = NUM_TASKS * CLASSES_POR_TAREFA

BATCH_SIZE = int(CFG_TREINO.get("batch_size", 48))
EPOCHS_POR_TAREFA = int(CFG_TREINO.get("epochs_por_tarefa", 100))
DIM_EMBEDDING = int(CFG_TREINO.get("embedding_dim", 512))
R_ADAPTER = int(CFG_TREINO.get("r_adapter", 64))
WEIGHT_DECAY_ADAPTER = float(CFG_TREINO.get("weight_decay_adapter", 1e-4))
LAMBDA_LVQ = float(CFG_TREINO.get("lambda_lvq", 0.3))
ALPHA_LVQ = float(CFG_TREINO.get("alpha_lvq", 0.4))
TAU_LVQ = float(CFG_TREINO.get("tau_lvq", 1.0))
UNIDADES_LVQ_POR_CLASSE = int(CFG_TREINO.get("unidades_lvq_por_classe", 16))
FREQ_LOG_EPOCAS = int(CFG_TREINO.get("frequencia_log_epocas", 10))

# [Estratégia 2] pseudo-negativos
CFG_PN = CFG.get("pseudo_negativos", {"modo": "nenhum"})
PN_MODO = str(CFG_PN.get("modo", "nenhum")).lower()
PN_POR_LOTE = int(CFG_PN.get("por_lote", BATCH_SIZE))
LAMBDA_PN = float(CFG_PN.get("lambda_pn", 0.5))

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# [Estratégia 2] identificador único da execução: <config>_seed<seed>_<data_hora>, por exemplo
# "baseline_seed1993_20261106_101500". Vai no nome do log, do JSON, do checkpoint e da pasta de
# figuras, para que as 8 execuções do rodar_experimentos.sh não sobrescrevam umas às outras.
NOME_CONFIG = os.path.splitext(os.path.basename(CFG.get("_arquivo_config", "config")))[0]
if NOME_CONFIG.startswith("config_"):
    NOME_CONFIG = NOME_CONFIG[len("config_"):]
ID_EXECUCAO = f"{NOME_CONFIG}_seed{SEED}_{timestamp}"

# [Estratégia 2] Diretórios de saída: DENTRO do projeto, em CIL_e_GFlowNet/resultados/, qualquer
# que seja a pasta de onde o comando é executado (a Estratégia 1 usava ../ relativo ao cwd).
OS_RESULTADOS_RAIZ = os.path.join(SCRIPT_DIR, "resultados")
OS_LOG_DIR = os.path.join(OS_RESULTADOS_RAIZ, "logs")
OS_RESULTS_DIR = os.path.join(OS_RESULTADOS_RAIZ, "results")
OS_FIGURES_DIR = os.path.join(OS_RESULTADOS_RAIZ, "figures_cil", ID_EXECUCAO)   # uma pasta por execução
OS_CHECKPOINTS_DIR = os.path.join(OS_RESULTADOS_RAIZ, "checkpoints")
for _d in (OS_LOG_DIR, OS_RESULTS_DIR, OS_FIGURES_DIR, OS_CHECKPOINTS_DIR):
    os.makedirs(_d, exist_ok=True)

log_filename = os.path.join(OS_LOG_DIR, f"experiment_{ID_EXECUCAO}.log")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_filename, encoding="utf-8"), logging.StreamHandler()]
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================================
# 1. REPRODUTIBILIDADE
# ============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)
logging.info(f"[Reprodutibilidade] Seed={SEED} (random, numpy, torch, cudnn determinístico). "
             f"Esta mesma seed também controla o sorteio das classes por tarefa.")
logging.info(f"[Configuração] dataset={CFG_DATASET['nome']} | num_tasks={NUM_TASKS} | "
             f"classes_por_tarefa={CLASSES_POR_TAREFA} | batch_size={BATCH_SIZE} | "
             f"otimizador={CFG_OTIMIZADOR.get('tipo', 'SGD')} | device={DEVICE} | "
             f"pseudo_negativos={PN_MODO}")


# ============================================================================
# 2. DATASET (plugável: cifar10 / cifar100 / omnibenchmark) + PARTIÇÃO DE
#    TAREFAS POR SORTEIO (seed definida no config.json)
# ============================================================================
def carregar_dataset_base(cfg_dataset):
    """Retorna (train_set, test_set), ambos expondo .targets (lista de
    inteiros) e .classes (lista de nomes), interface comum a
    torchvision.datasets.CIFAR10/CIFAR100 e torchvision.datasets.ImageFolder
    -- permite que todo o restante do pipeline seja agnóstico ao dataset."""
    nome = cfg_dataset["nome"].strip().lower()
    resize = tuple(cfg_dataset.get("resize", [64, 64]))

    # [Estratégia 2] caminhos relativos do config são resolvidos a partir da PASTA DO SCRIPT
    # (não da pasta de onde o comando é executado), para "../../my_data" sempre apontar para o
    # mesmo lugar. Barras "/" funcionam também no Windows.
    def _resolver(caminho):
        caminho = os.path.expanduser(caminho)
        return caminho if os.path.isabs(caminho) else os.path.normpath(os.path.join(SCRIPT_DIR, caminho))

    cfg_dataset = dict(cfg_dataset)
    for chave in ("raiz_cifar", "raiz_omnibenchmark"):
        if chave in cfg_dataset:
            cfg_dataset[chave] = _resolver(cfg_dataset[chave])
    if nome in ("cifar10", "cifar100"):
        pasta_esperada = "cifar-10-batches-py" if nome == "cifar10" else "cifar-100-python"
        raiz = cfg_dataset.get("raiz_cifar", _resolver("../data"))
        if os.path.isdir(os.path.join(raiz, pasta_esperada)):
            logging.info(f"[Dataset] Usando {nome} já baixado em {os.path.join(raiz, pasta_esperada)}")
        else:
            logging.warning(f"[Dataset] Pasta {pasta_esperada!r} NÃO encontrada em {raiz}. O torchvision vai "
                            f"tentar baixar o dataset para lá. Se ele já foi baixado, confira "
                            f"'dataset.raiz_cifar' em config_estrategia_2.json: deve apontar para a pasta que "
                            f"CONTÉM {pasta_esperada!r}, não para ela mesma.")
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    if nome == "cifar10":
        raiz = cfg_dataset.get("raiz_cifar", "../data")
        train_set = torchvision.datasets.CIFAR10(root=raiz, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root=raiz, train=False, download=True, transform=transform)
    elif nome == "cifar100":
        raiz = cfg_dataset.get("raiz_cifar", "../data")
        train_set = torchvision.datasets.CIFAR100(root=raiz, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR100(root=raiz, train=False, download=True, transform=transform)
    elif nome == "omnibenchmark":
        raiz = cfg_dataset["raiz_omnibenchmark"]
        raiz_treino = os.path.join(raiz, "train")
        raiz_teste = os.path.join(raiz, "test")
        if not os.path.isdir(raiz_treino) or not os.path.isdir(raiz_teste):
            raise FileNotFoundError(
                f"OmniBenchmark esperado em formato ImageFolder com subpastas 'train/' e 'test/' "
                f"dentro de {raiz!r} (cada uma com uma subpasta por classe). "
                f"Ajuste 'dataset.raiz_omnibenchmark' em config.json.")
        train_set = torchvision.datasets.ImageFolder(raiz_treino, transform=transform)
        test_set = torchvision.datasets.ImageFolder(raiz_teste, transform=transform)
    else:
        raise ValueError(f"Dataset não suportado: {nome!r} (use 'cifar10', 'cifar100' ou 'omnibenchmark')")

    logging.info(f"[Dataset] {nome} carregado — {len(train_set.classes)} classes disponíveis, "
                 f"{len(train_set)} imagens de treino, {len(test_set)} imagens de teste.")
    return train_set, test_set


def montar_particao_tarefas(num_classes_disponiveis, num_tasks, classes_por_tarefa, seed):
    """Sorteia, sem reposição, num_tasks*classes_por_tarefa classes do
    dataset (índices originais) e as divide em num_tasks blocos de
    classes_por_tarefa cada -- a ORDEM dos blocos define a ordem das tarefas
    (tarefa 0 = primeiro bloco sorteado, etc). Determinístico em função de
    'seed', conforme pedido: a seed fica inteiramente no config.json."""
    necessarias = num_tasks * classes_por_tarefa
    if necessarias > num_classes_disponiveis:
        raise ValueError(
            f"num_tasks*classes_por_tarefa={necessarias} excede as {num_classes_disponiveis} "
            f"classes disponíveis no dataset escolhido.")
    rng = np.random.RandomState(seed)
    classes_sorteadas = rng.choice(num_classes_disponiveis, size=necessarias, replace=False)
    tarefas_classes_originais = [
        classes_sorteadas[i * classes_por_tarefa:(i + 1) * classes_por_tarefa].tolist()
        for i in range(num_tasks)
    ]
    return tarefas_classes_originais


def remapear_rotulos(dataset, mapa_original_para_cil):
    """Remapeia rótulos originais do dataset para índices CIL contíguos
    (0..NUM_CLASSES_TOTAIS-1), em ordem (classes de índice CIL mais alto
    pertencem a tarefas posteriores). Trata tanto datasets CIFAR (.targets é
    a lista usada diretamente por __getitem__) quanto ImageFolder (.samples
    é o que __getitem__ realmente usa; .targets é só espelho de leitura)."""
    if hasattr(dataset, "samples"):
        dataset.samples = [(caminho, mapa_original_para_cil.get(rotulo, rotulo))
                            for caminho, rotulo in dataset.samples]
        dataset.targets = [rotulo for _, rotulo in dataset.samples]
        if hasattr(dataset, "imgs"):
            dataset.imgs = dataset.samples
    else:
        for idx in range(len(dataset.targets)):
            rotulo = dataset.targets[idx]
            if rotulo in mapa_original_para_cil:
                dataset.targets[idx] = mapa_original_para_cil[rotulo]


def preparar_particao_e_nomes(train_set, tarefas_classes_originais):
    mapa_original_para_cil = {}
    nomes_classe_cil = {}
    classe_cil = 0
    for tarefa_orig in tarefas_classes_originais:
        for c_orig in tarefa_orig:
            mapa_original_para_cil[c_orig] = classe_cil
            nomes_classe_cil[classe_cil] = str(train_set.classes[c_orig])
            classe_cil += 1
    return mapa_original_para_cil, nomes_classe_cil


def obter_loaders_cil(train_set, test_set, tarefas_classes_originais, mapa_original_para_cil, batch_size):
    """Generaliza obter_loaders_cil() da Estratégia 26 para N tarefas de
    tamanho arbitrário. Os índices por tarefa/por classe são calculados
    ANTES do remapeamento de rótulos (usando os rótulos originais do
    dataset), e só então os rótulos são remapeados para o espaço CIL — a
    mesma ordem de operações da Estratégia 26, generalizada."""
    targets_treino_originais = list(train_set.targets)
    targets_teste_originais = list(test_set.targets)

    indices_treino_por_tarefa = [
        [i for i, l in enumerate(targets_treino_originais) if l in tarefa]
        for tarefa in tarefas_classes_originais
    ]
    indices_teste_por_tarefa = [
        [i for i, l in enumerate(targets_teste_originais) if l in tarefa]
        for tarefa in tarefas_classes_originais
    ]
    indices_teste_por_classe_cil = {}
    classe_cil = 0
    for tarefa_orig in tarefas_classes_originais:
        for c_orig in tarefa_orig:
            indices_teste_por_classe_cil[classe_cil] = [i for i, l in enumerate(targets_teste_originais) if l == c_orig]
            classe_cil += 1

    remapear_rotulos(train_set, mapa_original_para_cil)
    remapear_rotulos(test_set, mapa_original_para_cil)

    loaders_treino = [DataLoader(Subset(train_set, idx), batch_size=batch_size, shuffle=True)
                      for idx in indices_treino_por_tarefa]
    loaders_teste_tarefa = [DataLoader(Subset(test_set, idx), batch_size=batch_size, shuffle=False)
                             for idx in indices_teste_por_tarefa]
    loaders_teste_classe = {c: DataLoader(Subset(test_set, idx), batch_size=batch_size, shuffle=False)
                             for c, idx in indices_teste_por_classe_cil.items()}

    loaders_teste_acumulado = []
    indices_acumulados = []
    for idx_tarefa in indices_teste_por_tarefa:
        indices_acumulados = indices_acumulados + idx_tarefa
        loaders_teste_acumulado.append(
            DataLoader(Subset(test_set, list(indices_acumulados)), batch_size=batch_size, shuffle=False))

    return loaders_treino, loaders_teste_tarefa, loaders_teste_acumulado, loaders_teste_classe


# ============================================================================
# 3. BACKBONE PRÉ-TREINADO, CONGELADO DESDE A CRIAÇÃO (idêntico à Estratégia 26)
# ============================================================================
def criar_backbone_pretreinado_congelado():
    try:
        pesos = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        backbone = torchvision.models.resnet18(weights=pesos)
    except AttributeError:
        backbone = torchvision.models.resnet18(pretrained=True)
    backbone.fc = nn.Identity()
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    logging.info("[Backbone] ResNet-18 pré-treinada no ImageNet, congelada desde a criação.")
    return backbone.to(DEVICE)


def extrair_embedding(backbone, x):
    with torch.no_grad():
        return backbone(x)


# ============================================================================
# 4. ADAPTER (bottleneck, idêntico às Estratégias 24-27c)
# ============================================================================
class Adapter(nn.Module):
    def __init__(self, dim=DIM_EMBEDDING, r=R_ADAPTER):
        super().__init__()
        self.down = nn.Linear(dim, r)
        self.up = nn.Linear(r, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        h = F.relu(self.down(x))
        return self.up(h)


def congelar_adapter(adapter):
    for p in adapter.parameters():
        p.requires_grad = False
    adapter.eval()


def construir_adapter_aleatorio_generico(seed_diferente, dim=DIM_EMBEDDING, r=R_ADAPTER):
    """Adapter NUNCA treinado, com down E up inicializados aleatoriamente e
    não-degenerados (diferente do init padrão do projeto, que zera 'up') --
    usado como controle no Teste C do diagnóstico A/B/C."""
    g = torch.Generator(device='cpu').manual_seed(seed_diferente)
    adapter_aleatorio = Adapter(dim=dim, r=r)
    with torch.no_grad():
        limite_down = 1.0 / (dim ** 0.5)
        limite_up = 1.0 / (r ** 0.5)
        adapter_aleatorio.down.weight.copy_((torch.rand(r, dim, generator=g) * 2 - 1) * limite_down)
        adapter_aleatorio.down.bias.copy_((torch.rand(r, generator=g) * 2 - 1) * limite_down)
        adapter_aleatorio.up.weight.copy_((torch.rand(dim, r, generator=g) * 2 - 1) * limite_up)
        adapter_aleatorio.up.bias.copy_((torch.rand(dim, generator=g) * 2 - 1) * limite_up)
    adapter_aleatorio = adapter_aleatorio.to(DEVICE)
    adapter_aleatorio.eval()
    for p in adapter_aleatorio.parameters():
        p.requires_grad = False
    return adapter_aleatorio


# ============================================================================
# 5. OTIMIZADOR CONFIGURÁVEL (config.json: otimizador.*) — SGD por default
# ============================================================================
def construir_otimizador(param_groups, cfg_otimizador):
    tipo = str(cfg_otimizador.get("tipo", "SGD")).strip().upper()
    if tipo == "SGD":
        return optim.SGD(
            param_groups,
            lr=cfg_otimizador.get("lr", 0.01),
            momentum=cfg_otimizador.get("momentum", 0.9),
            weight_decay=cfg_otimizador.get("weight_decay", 0.0),
            nesterov=cfg_otimizador.get("nesterov", False),
        )
    elif tipo == "ADAM":
        return optim.Adam(
            param_groups,
            lr=cfg_otimizador.get("lr", 1e-4),
            betas=tuple(cfg_otimizador.get("betas", [0.9, 0.999])),
            weight_decay=cfg_otimizador.get("weight_decay", 0.0),
        )
    elif tipo == "ADAMW":
        return optim.AdamW(
            param_groups,
            lr=cfg_otimizador.get("lr", 1e-4),
            betas=tuple(cfg_otimizador.get("betas", [0.9, 0.999])),
            weight_decay=cfg_otimizador.get("weight_decay", 1e-2),
        )
    elif tipo == "RMSPROP":
        return optim.RMSprop(
            param_groups,
            lr=cfg_otimizador.get("lr", 1e-3),
            momentum=cfg_otimizador.get("momentum", 0.0),
            weight_decay=cfg_otimizador.get("weight_decay", 0.0),
        )
    else:
        raise ValueError(f"Otimizador não suportado: {tipo!r} (use 'SGD', 'Adam', 'AdamW' ou 'RMSprop')")


# ============================================================================
# 6. FC EXPANSÍVEL (Gram-Schmidt, idêntico às Estratégias 24-27c)
# ============================================================================
def criar_fc_ortogonal_inicial(num_classes, dim=DIM_EMBEDDING):
    fc = torch.empty(num_classes, dim)
    nn.init.orthogonal_(fc)
    return fc.to(DEVICE)


def expandir_fc_ortogonal(fc_atual, n_novas, dim=DIM_EMBEDDING, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    vetores = [fc_atual[i] for i in range(fc_atual.shape[0])]
    novos = []
    for _ in range(n_novas):
        v = torch.randn(dim, generator=g).to(DEVICE)
        for w in vetores:
            w_norm = w / torch.norm(w)
            v = v - torch.dot(v, w_norm) * w_norm
        v = v / torch.norm(v)
        vetores.append(v)
        novos.append(v)
    return torch.cat([fc_atual, torch.stack(novos, dim=0)], dim=0)


# ============================================================================
# 7. PROTÓTIPOS — média da saída do Adapter da própria tarefa (idêntico)
# ============================================================================
def calcular_prototipo(backbone, adapter, loader, classe_alvo, dim=DIM_EMBEDDING):
    soma = torch.zeros(dim, device=DEVICE)
    contagem = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            phi = backbone(images)
            emb = adapter(phi)
            mask = labels == classe_alvo
            if mask.any():
                soma += emb[mask].sum(dim=0)
                contagem += int(mask.sum().item())
    prototipo = soma / contagem
    prototipo = prototipo / torch.norm(prototipo)
    return prototipo, contagem


# ============================================================================
# 8. MAPA HEXAGONAL LVQ — generaliza a grade fixa 12x8/3x2 da Estratégia 26
#    para num_tasks/classes_por_tarefa/unidades_lvq_por_classe arbitrários
# ============================================================================
def _par_divisores_mais_quadrado(n):
    """Entre todos os pares (a,b) com a*b=n, devolve o mais próximo de um
    quadrado (menor |a-b|) -- para n=16 devolve (4,4), reproduzindo
    exatamente o bloco 4x4 da Estratégia 26."""
    melhor = (1, n)
    menor_diferenca = n - 1
    for d in range(1, int(math.isqrt(n)) + 1):
        if n % d == 0:
            par = (d, n // d)
            diferenca = abs(par[0] - par[1])
            if diferenca < menor_diferenca:
                menor_diferenca = diferenca
                melhor = par
    return melhor


def dimensionar_grade_lvq(num_tasks, classes_por_tarefa, unidades_por_classe):
    """linhas = num_tasks * bloco_altura; colunas = classes_por_tarefa *
    bloco_largura -- reduz-se exatamente à grade 12x8 (3 blocos de linha x 2
    de coluna, 4x4=16 unidades/classe) da Estratégia 26 quando num_tasks=3,
    classes_por_tarefa=2, unidades_por_classe=16."""
    bloco_altura, bloco_largura = _par_divisores_mais_quadrado(unidades_por_classe)
    linhas = num_tasks * bloco_altura
    colunas = classes_por_tarefa * bloco_largura
    return linhas, colunas


def _centro_hex_flat_top(col, row, tamanho):
    dx = 1.5 * tamanho
    dy = np.sqrt(3) * tamanho
    x = col * dx
    y = row * dy + (dy / 2 if col % 2 else 0)
    return x, y


def _classe_da_unidade(col, row, linhas, colunas, blocos_linha, blocos_coluna):
    """Partição fixa em blocos_linha x blocos_coluna blocos -- idêntica à
    Estratégia 26 (lá, blocos_linha=num_tasks=3, blocos_coluna=
    classes_por_tarefa=2); aqui blocos_linha/blocos_coluna são sempre
    num_tasks/classes_por_tarefa, generalizando sem alterar a regra."""
    tam_bloco_linha = linhas // blocos_linha
    tam_bloco_coluna = colunas // blocos_coluna
    bloco_linha = min(row // tam_bloco_linha, blocos_linha - 1)
    bloco_coluna = min(col // tam_bloco_coluna, blocos_coluna - 1)
    return bloco_linha * blocos_coluna + bloco_coluna


class MapaLVQ(nn.Module):
    def __init__(self, linhas, colunas, dim, num_classes, blocos_linha, blocos_coluna):
        super().__init__()
        self.linhas = linhas
        self.colunas = colunas
        self.num_unidades = linhas * colunas
        self.num_classes = num_classes
        self.prototipos = nn.Parameter(torch.randn(self.num_unidades, dim) * 0.01)

        coords = [(c, r) for c in range(colunas) for r in range(linhas)]
        self.coords_grade = coords
        self.unidade_para_classe = [
            _classe_da_unidade(c, r, linhas, colunas, blocos_linha, blocos_coluna) for (c, r) in coords
        ]
        self.indices_por_classe = {
            k: [i for i, cl in enumerate(self.unidade_para_classe) if cl == k]
            for k in range(num_classes)
        }
        mascara_classe = np.zeros((num_classes, self.num_unidades), dtype=np.float32)
        for i, cl in enumerate(self.unidade_para_classe):
            mascara_classe[cl, i] = 1.0
        self.register_buffer("mascara_por_classe", torch.tensor(mascara_classe))
        self.register_buffer("mascara_treinavel", torch.zeros(self.num_unidades))
        self.register_buffer("contagem_hits", torch.zeros(self.num_unidades))
        self.hits_por_classe = {}

    def prototipos_efetivos(self):
        m = self.mascara_treinavel.unsqueeze(1)
        return self.prototipos * m + self.prototipos.detach() * (1 - m)

    def ativar_classes(self, classes):
        with torch.no_grad():
            for c in classes:
                for i in self.indices_por_classe[c]:
                    self.mascara_treinavel[i] = 1.0

    def congelar_classes(self, classes):
        with torch.no_grad():
            for c in classes:
                for i in self.indices_por_classe[c]:
                    self.mascara_treinavel[i] = 0.0

    def perda_lvq(self, z, labels, classes_ativas):
        protos = self.prototipos_efetivos()
        d2 = torch.cdist(z, protos) ** 2
        mask_certos = self.mascara_por_classe[labels]
        d2_certos = d2.masked_fill(mask_certos == 0, float('inf'))
        d_mais, bmu_certo = d2_certos.min(dim=1)

        mask_classes_ativas = self.mascara_por_classe[classes_ativas].sum(dim=0).clamp(max=1)
        mask_errados = mask_classes_ativas.unsqueeze(0) * (1 - mask_certos)
        d2_errados = d2.masked_fill(mask_errados == 0, float('inf'))
        d_menos, _ = d2_errados.min(dim=1)

        mu = (d_mais - d_menos) / (d_mais + d_menos + 1e-8)
        return mu.mean(), bmu_certo

    def atualizar_contagem_hits(self, bmu, labels):
        with torch.no_grad():
            for u in bmu.unique():
                self.contagem_hits[u] += (bmu == u).sum().float()
        bmu_np = bmu.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        for u_val, l_val in zip(bmu_np, labels_np):
            if l_val not in self.hits_por_classe:
                self.hits_por_classe[l_val] = np.zeros(self.num_unidades)
            self.hits_por_classe[l_val][u_val] += 1

    def score_lvq(self, z, classe, tau):
        idx = self.indices_por_classe[classe]
        d2 = torch.cdist(z, self.prototipos[idx]) ** 2
        d_min = d2.min(dim=1).values
        return torch.exp(-d_min / (2 * tau ** 2))


def criar_mapa_lvq(linhas, colunas, dim, num_classes, blocos_linha, blocos_coluna):
    return MapaLVQ(linhas=linhas, colunas=colunas, dim=dim, num_classes=num_classes,
                    blocos_linha=blocos_linha, blocos_coluna=blocos_coluna).to(DEVICE)


# ============================================================================
# 9. CLASSIFICAÇÃO — score combinado (cosseno+LVQ), generaliza
#    calcular_logits_finais_4vias/6vias da Estratégia 26 para N classes/N
#    Adapters: a classe c é servida pelo Adapter da tarefa c // classes_por_tarefa
# ============================================================================
def calcular_logits_finais(backbone, adapters, fc, lvq, images, classes_por_tarefa, alpha_lvq, tau_lvq):
    phi = backbone(images)
    deltas_por_tarefa = [adapter(phi) for adapter in adapters]
    num_classes_ativas = fc.shape[0]
    scores = []
    for c in range(num_classes_ativas):
        tarefa_idx = c // classes_por_tarefa
        delta_c = deltas_por_tarefa[tarefa_idx]
        s_cos = F.cosine_similarity(delta_c, fc[c].unsqueeze(0).expand_as(delta_c), dim=1)
        s_lvq = lvq.score_lvq(delta_c, c, tau_lvq)
        scores.append((1 - alpha_lvq) * s_cos + alpha_lvq * s_lvq)
    return torch.stack(scores, dim=1)


def testar_por_classe(backbone, adapters, fc, lvq, loader_teste, classes_avaliadas,
                       classes_por_tarefa, alpha_lvq, tau_lvq):
    corretos = {c: 0 for c in classes_avaliadas}
    total = {c: 0 for c in classes_avaliadas}
    with torch.no_grad():
        for images, labels in loader_teste:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = calcular_logits_finais(backbone, adapters, fc, lvq, images, classes_por_tarefa, alpha_lvq, tau_lvq)
            preds = logits.argmax(dim=1)
            for c in classes_avaliadas:
                mask = labels == c
                total[c] += int(mask.sum().item())
                corretos[c] += int((preds[mask] == labels[mask]).sum().item())
    return {c: (100.0 * corretos[c] / total[c] if total[c] > 0 else 0.0) for c in classes_avaliadas}


def prever_final(backbone, adapters, fc, lvq, loader_teste, classes_por_tarefa, alpha_lvq, tau_lvq):
    todos_labels, todos_preds = [], []
    with torch.no_grad():
        for images, labels in loader_teste:
            images = images.to(DEVICE)
            logits = calcular_logits_finais(backbone, adapters, fc, lvq, images, classes_por_tarefa, alpha_lvq, tau_lvq)
            preds = logits.argmax(dim=1).cpu().numpy()
            todos_preds.append(preds)
            todos_labels.append(labels.numpy())
    return np.concatenate(todos_labels), np.concatenate(todos_preds)


# ============================================================================
# 10. TREINO DE UMA TAREFA — generaliza treinar_task0/task1/task2 da
#     Estratégia 26 em uma única função parametrizada pelo índice da tarefa.
#     Tarefa 0 é apenas o caso particular "sem adapters anteriores".
# ============================================================================
def treinar_tarefa(indice_tarefa, backbone, lvq, adapters_anteriores, fc_atual,
                    loader_treino, loader_teste_acumulado, nomes_classe_cil, gerador=None):
    classes_novas = list(range(indice_tarefa * CLASSES_POR_TAREFA, (indice_tarefa + 1) * CLASSES_POR_TAREFA))
    todas_classes_ativas = list(range(0, (indice_tarefa + 1) * CLASSES_POR_TAREFA))
    nomes_tarefa = ", ".join(nomes_classe_cil[c] for c in classes_novas)

    if indice_tarefa == 0:
        fc_expandida = criar_fc_ortogonal_inicial(num_classes=CLASSES_POR_TAREFA, dim=DIM_EMBEDDING)
    else:
        fc_expandida = expandir_fc_ortogonal(fc_atual, n_novas=CLASSES_POR_TAREFA, dim=DIM_EMBEDDING,
                                              seed=SEED + 100 + indice_tarefa)

    adapter_novo = Adapter(dim=DIM_EMBEDDING, r=R_ADAPTER).to(DEVICE)
    theta_init_adapter = copy.deepcopy(adapter_novo.state_dict())
    theta_init_lvq = lvq.prototipos.detach().clone()
    logging.info(f"[Tarefa {indice_tarefa}] Snapshots de θ_init (Adapter_{indice_tarefa} e unidades LVQ "
                 f"da tarefa) capturados antes do treino (diagnóstico de negação).")

    lvq.ativar_classes(classes_novas)
    optimizer = construir_otimizador(
        [{'params': adapter_novo.parameters(), 'weight_decay': WEIGHT_DECAY_ADAPTER},
         {'params': lvq.parameters(), 'weight_decay': 0.0}],
        CFG_OTIMIZADOR)
    criterion = nn.CrossEntropyLoss()

    logging.info(f"=== Tarefa {indice_tarefa} ({nomes_tarefa}): treinando Adapter_{indice_tarefa} "
                 f"(PERMANENTE) + unidades LVQ da tarefa ===")
    # [Estratégia 2] pseudo-negativos só existem a partir da tarefa 1 (precisam de classes antigas)
    usar_pn = gerador is not None and indice_tarefa >= 1 and PN_MODO != "nenhum"
    if usar_pn:
        logging.info(f"[Tarefa {indice_tarefa}] Pseudo-negativos ATIVOS: modo={PN_MODO}, "
                     f"{PN_POR_LOTE} por lote, lambda_pn={LAMBDA_PN}, classes antigas="
                     f"{[nomes_classe_cil[c] for c in gerador.classes_antigas]}")

    historico = []
    for epoch in range(EPOCHS_POR_TAREFA):
        adapter_novo.train()
        running_loss_ce, running_loss_lvq, corretos, total = 0.0, 0.0, 0, 0
        running_loss_pn, total_pn = 0.0, 0
        if usar_pn:
            gerador.reiniciar_estatisticas()
        for images, labels in loader_treino:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            phi = extrair_embedding(backbone, images)
            with torch.no_grad():
                deltas_anteriores = [ad(phi) for ad in adapters_anteriores]
            delta_novo = adapter_novo(phi)
            deltas_todos = deltas_anteriores + [delta_novo]

            logits_cos = torch.stack([
                F.cosine_similarity(deltas_todos[c // CLASSES_POR_TAREFA],
                                     fc_expandida[c].unsqueeze(0).expand_as(deltas_todos[c // CLASSES_POR_TAREFA]),
                                     dim=1)
                for c in todas_classes_ativas
            ], dim=1)
            loss_ce = criterion(logits_cos, labels)
            loss_lvq, bmu = lvq.perda_lvq(delta_novo, labels, classes_ativas=todas_classes_ativas)
            loss = (1 - LAMBDA_LVQ) * loss_ce + LAMBDA_LVQ * loss_lvq

            # [Estratégia 2] phi sintéticos de classes antigas, com rótulo antigo. São avaliados
            # como na inferência (Adapters antigos congelados para as classes antigas, Adapter_t
            # para as novas, cosseno + LVQ). Só o Adapter_t e as unidades LVQ da tarefa t recebem
            # gradiente: a única forma de acertar o rótulo antigo é baixar o score das classes novas.
            if usar_pn:
                ctx = {"adapters_anteriores": adapters_anteriores, "adapter_novo": adapter_novo,
                       "fc": fc_expandida, "lvq": lvq, "classes_ativas": todas_classes_ativas,
                       "classes_novas": classes_novas}
                phi_pn, y_pn = gerador.amostrar(PN_POR_LOTE, ctx)
                S_pn = scores_combinados(phi_pn, adapters_anteriores, adapter_novo, fc_expandida, lvq,
                                         todas_classes_ativas, CLASSES_POR_TAREFA, ALPHA_LVQ, TAU_LVQ,
                                         grad_novo=True)
                loss_pn = criterion(S_pn, y_pn)
                loss = loss + LAMBDA_PN * loss_pn
                running_loss_pn += loss_pn.item() * len(y_pn)
                total_pn += len(y_pn)

            loss.backward()
            optimizer.step()
            lvq.atualizar_contagem_hits(bmu, labels)

            running_loss_ce += loss_ce.item() * images.size(0)
            running_loss_lvq += loss_lvq.item() * images.size(0)
            _, predicted = logits_cos.max(1)
            total += labels.size(0)
            corretos += predicted.eq(labels).sum().item()

        loss_ce_treino = running_loss_ce / total
        loss_lvq_treino = running_loss_lvq / total
        acc_treino = (corretos / total) * 100

        adapter_novo.eval()
        adapters_atuais = adapters_anteriores + [adapter_novo]
        acc_por_classe = testar_por_classe(backbone, adapters_atuais, fc_expandida, lvq, loader_teste_acumulado,
                                            todas_classes_ativas, CLASSES_POR_TAREFA, ALPHA_LVQ, TAU_LVQ)
        acc_por_tarefa_vista = [
            float(np.mean([acc_por_classe[c] for c in range(t * CLASSES_POR_TAREFA, (t + 1) * CLASSES_POR_TAREFA)]))
            for t in range(indice_tarefa + 1)
        ]
        aac_ep = float(np.mean(acc_por_tarefa_vista))

        registro = {
            "epoca": epoch + 1, "loss_ce": loss_ce_treino, "loss_lvq": loss_lvq_treino,
            "acc_treino": acc_treino, "acc_por_classe": {nomes_classe_cil[c]: acc_por_classe[c]
                                                          for c in todas_classes_ativas},
            "acc_por_tarefa": acc_por_tarefa_vista, "aac": aac_ep,
        }
        if usar_pn:
            # [Estratégia 2] diagnósticos por época:
            #   frac_vaza  = fração dos pseudo-negativos com margem > 0 (vazariam para classe nova);
            #   ess_frac   = tamanho efetivo da amostragem por importância / nº de propostas
            #                (perto de 0 = as regiões de vazamento estão na cauda das gaussianas);
            #   logZ_is vs logZ_gfn = se a GFlowNet fica abaixo, está deixando massa para trás.
            registro["loss_pn"] = running_loss_pn / max(total_pn, 1)
            registro["pseudo_negativos"] = gerador.resumo_estatisticas()
        historico.append(registro)

        if (epoch + 1) % FREQ_LOG_EPOCAS == 0 or epoch == EPOCHS_POR_TAREFA - 1:
            resumo_classes = " | ".join(f"{nomes_classe_cil[c]}: {acc_por_classe[c]:.2f}%" for c in classes_novas)
            logging.info(f"   [Tarefa {indice_tarefa}] Época [{epoch+1}/{EPOCHS_POR_TAREFA}] -> "
                         f"CE:{loss_ce_treino:.4f} | L_LVQ:{loss_lvq_treino:.4f} | "
                         f"Acc treino: {acc_treino:.2f}% | AAC acumulado: {aac_ep:.2f}% | {resumo_classes}")
            if usar_pn:
                st = registro["pseudo_negativos"]
                fmt = lambda v, f: "-" if v is None else format(v, f)
                logging.info(f"      [Pseudo-negativos] CE_pn:{registro['loss_pn']:.4f} | "
                             f"vazam (margem>0): {fmt(st['frac_vaza'], '.1%')} | "
                             f"margem média: {fmt(st['margem_media'], '+.4f')} | "
                             f"ESS/propostas: {fmt(st['ess_frac'], '.1%')} | "
                             f"logZ IS: {fmt(st['logZ_is'], '.3f')} | logZ GFN: {fmt(st['logZ_gfn'], '.3f')} | "
                             f"perda TB: {fmt(st['perda_tb'], '.2f')}")

    congelar_adapter(adapter_novo)
    lvq.congelar_classes(classes_novas)
    logging.info(f"[Tarefa {indice_tarefa}] Adapter_{indice_tarefa} CONGELADO e MANTIDO; unidades LVQ da "
                 f"tarefa CONGELADAS (AAC diagnóstica final: {historico[-1]['aac']:.2f}%).")

    prototipos_novos = []
    for c in classes_novas:
        w, n = calcular_prototipo(backbone, adapter_novo, loader_treino, classe_alvo=c, dim=DIM_EMBEDDING)
        prototipos_novos.append(w)
        logging.info(f"[Tarefa {indice_tarefa}] Protótipo (FC) de {nomes_classe_cil[c]} substituído "
                     f"({n} imagens de treino).")

    if indice_tarefa == 0:
        fc_final = torch.stack(prototipos_novos, dim=0)
    else:
        fc_final = torch.cat([fc_expandida[:indice_tarefa * CLASSES_POR_TAREFA]] +
                              [p.unsqueeze(0) for p in prototipos_novos], dim=0)

    return adapter_novo, fc_final, historico, theta_init_adapter, theta_init_lvq, classes_novas


# ============================================================================
# 11. DIAGNÓSTICO A/B/C — TASK ARITHMETIC (Ilharco et al., ICLR 2023),
#     generalizado para qualquer fronteira tarefa_t -> tarefa_{t+1}
#     (a Estratégia 26 só cobria duas fronteiras fixas, M1->M2 e M2->M3)
# ============================================================================
def construir_versao_negada(adapter_treinado, theta_init_adapter, lvq_treinado, theta_init_lvq,
                             indices_classes_tarefa, linhas, colunas, blocos_linha, blocos_coluna):
    adapter_negado = Adapter(dim=DIM_EMBEDDING, r=R_ADAPTER).to(DEVICE)
    theta_treinado = adapter_treinado.state_dict()
    theta_negado = {k: 2 * theta_init_adapter[k] - theta_treinado[k] for k in theta_treinado}
    adapter_negado.load_state_dict(theta_negado)
    adapter_negado.eval()
    for p in adapter_negado.parameters():
        p.requires_grad = False

    lvq_negado = criar_mapa_lvq(linhas, colunas, DIM_EMBEDDING, NUM_CLASSES_TOTAIS, blocos_linha, blocos_coluna)
    lvq_negado.load_state_dict(lvq_treinado.state_dict())
    with torch.no_grad():
        prototipos_negados = lvq_treinado.prototipos.detach().clone()
        prototipos_negados[indices_classes_tarefa] = (
            2 * theta_init_lvq[indices_classes_tarefa] - prototipos_negados[indices_classes_tarefa])
        lvq_negado.prototipos.copy_(prototipos_negados)
    lvq_negado.eval()
    return adapter_negado, lvq_negado


def construir_versao_aleatoria_lvq(lvq_treinado, indices_classes_tarefa, seed_diferente,
                                    linhas, colunas, blocos_linha, blocos_coluna):
    lvq_aleatorio = criar_mapa_lvq(linhas, colunas, DIM_EMBEDDING, NUM_CLASSES_TOTAIS, blocos_linha, blocos_coluna)
    lvq_aleatorio.load_state_dict(lvq_treinado.state_dict())
    g = torch.Generator(device='cpu').manual_seed(seed_diferente + 1)
    with torch.no_grad():
        prototipos_aleatorios = lvq_treinado.prototipos.detach().clone()
        novos = (torch.randn(len(indices_classes_tarefa), DIM_EMBEDDING, generator=g) * 0.01).to(DEVICE)
        prototipos_aleatorios[indices_classes_tarefa] = novos
        lvq_aleatorio.prototipos.copy_(prototipos_aleatorios)
    lvq_aleatorio.eval()
    return lvq_aleatorio


def avaliar_exclusao(backbone, adapters_anteriores, fc_ate_fronteira, lvq, loader_teste_ate_fronteira):
    corretos, total = 0, 0
    with torch.no_grad():
        for images, labels in loader_teste_ate_fronteira:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = calcular_logits_finais(backbone, adapters_anteriores, fc_ate_fronteira, lvq, images,
                                             CLASSES_POR_TAREFA, ALPHA_LVQ, TAU_LVQ)
            _, predicted = logits.max(1)
            total += labels.size(0)
            corretos += predicted.eq(labels).sum().item()
    return (corretos / total) * 100 if total > 0 else 0.0


def avaliar_com_substituicao(backbone, adapters_com_substituicao, fc, lvq_com_substituicao, loader_teste):
    corretos, total = 0, 0
    with torch.no_grad():
        for images, labels in loader_teste:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = calcular_logits_finais(backbone, adapters_com_substituicao, fc, lvq_com_substituicao, images,
                                             CLASSES_POR_TAREFA, ALPHA_LVQ, TAU_LVQ)
            _, predicted = logits.max(1)
            total += labels.size(0)
            corretos += predicted.eq(labels).sum().item()
    return (corretos / total) * 100 if total > 0 else 0.0


def diagnostico_task_arithmetic_fronteira(indice_fronteira, backbone, adapters_antigos, fc, lvq,
                                           adapter_novo, theta_init_adapter, theta_init_lvq, classes_novas,
                                           loader_teste_antigas, linhas, colunas, blocos_linha, blocos_coluna):
    """Testes A (negação), B (exclusão) e C (aleatório) no limite entre as
    tarefas 0..indice_fronteira (antigas, 'congeladas') e a tarefa
    indice_fronteira+1 recém-treinada (adapter_novo) -- generaliza
    construir_versao_negada_m2/m3 + avaliar_cat_dog_exclusao/
    avaliar_m1_m2_exclusao + construir_adapter_aleatorio_generico da
    Estratégia 26 para uma fronteira arbitrária."""
    fc_antigas = fc[:len(adapters_antigos) * CLASSES_POR_TAREFA]

    adapter_negado, lvq_negado = construir_versao_negada(
        adapter_novo, theta_init_adapter, lvq, theta_init_lvq, classes_novas, linhas, colunas, blocos_linha, blocos_coluna)
    acc_negacao = avaliar_com_substituicao(backbone, adapters_antigos + [adapter_negado], fc_antigas,
                                            lvq_negado, loader_teste_antigas)

    acc_exclusao = avaliar_exclusao(backbone, adapters_antigos, fc_antigas, lvq, loader_teste_antigas)

    seed_aleatorio = SEED + 10000 + 1000 * (indice_fronteira + 1)
    adapter_aleatorio = construir_adapter_aleatorio_generico(seed_aleatorio, dim=DIM_EMBEDDING, r=R_ADAPTER)
    lvq_aleatorio = construir_versao_aleatoria_lvq(lvq, classes_novas, seed_aleatorio, linhas, colunas,
                                                    blocos_linha, blocos_coluna)
    acc_aleatorio = avaliar_com_substituicao(backbone, adapters_antigos + [adapter_aleatorio], fc_antigas,
                                              lvq_aleatorio, loader_teste_antigas)

    acc_original = avaliar_com_substituicao(backbone, adapters_antigos + [adapter_novo], fc_antigas, lvq,
                                              loader_teste_antigas)

    if abs(acc_aleatorio - acc_exclusao) < 5 and abs(acc_negacao - acc_exclusao) < 5:
        interpretacao = ("Negação, Exclusão e versão aleatória convergem para o mesmo patamar -> o vazamento "
                          "é produto do TREINO da tarefa nova, não de uma direção 'anti' aprendida pela negação.")
    elif acc_aleatorio < acc_exclusao - 10:
        interpretacao = ("A versão aleatória ainda causa vazamento parcial -> sugere viés geométrico de fundo "
                          "independente do conteúdo aprendido.")
    else:
        interpretacao = ("Negação recupera acurácia de forma parecida à aleatória -> há interferência de "
                          "peso/conteúdo real, mas não distinguível de uma versão genérica não-treinada.")

    logging.info(f" > [Fronteira {indice_fronteira}->{indice_fronteira+1}][Original]  Acc tarefas antigas: {acc_original:.2f}%")
    logging.info(f" > [Fronteira {indice_fronteira}->{indice_fronteira+1}][Negação]   Acc tarefas antigas: {acc_negacao:.2f}%")
    logging.info(f" > [Fronteira {indice_fronteira}->{indice_fronteira+1}][Exclusão]  Acc tarefas antigas: {acc_exclusao:.2f}%")
    logging.info(f" > [Fronteira {indice_fronteira}->{indice_fronteira+1}][Aleatório] Acc tarefas antigas: {acc_aleatorio:.2f}%")
    logging.info(f" > Interpretação automática: {interpretacao}")

    return {
        "acc_original": acc_original, "acc_negacao": acc_negacao,
        "acc_exclusao": acc_exclusao, "acc_aleatorio": acc_aleatorio,
        "interpretacao_automatica": interpretacao,
    }


# ============================================================================
# 12. DRIFT SEMÂNTICO EXATO (Yu et al., 2020, p.5, Eq.8), generalizado para
#     qualquer número de tarefas/fronteiras
# ============================================================================
def calcular_drift_semantico(backbone, adapter, loader_treino_classe, classe_alvo, prototipo_congelado,
                              dim=DIM_EMBEDDING):
    prototipo_atual, n = calcular_prototipo(backbone, adapter, loader_treino_classe, classe_alvo, dim=dim)
    delta = prototipo_atual - prototipo_congelado
    norma_delta = float(torch.norm(delta).item())
    similaridade_cos = float(F.cosine_similarity(
        prototipo_congelado.unsqueeze(0), prototipo_atual.unsqueeze(0)).item())
    return norma_delta, similaridade_cos, n


# ============================================================================
# 13. NC1, MATRIZ DE SCORES MÉDIOS E MARGEM DE VAZAMENTO (generalizados)
# ============================================================================
def _nc1_de_embeddings(embeddings_por_classe):
    centroides = {}
    todos = []
    for k, lst in list(embeddings_por_classe.items()):
        if len(lst) > 0:
            arr = np.array(lst)
            embeddings_por_classe[k] = arr
            centroides[k] = np.mean(arr, axis=0)
            todos.append(arr)
    K = len(centroides)
    if K < 2:
        return float('inf')

    centroide_global = np.mean(np.concatenate(todos, axis=0), axis=0)
    D = len(centroide_global)
    sigma_w = np.zeros((D, D))
    N_total = 0
    for k in centroides:
        diff = embeddings_por_classe[k] - centroides[k]
        sigma_w += diff.T @ diff
        N_total += embeddings_por_classe[k].shape[0]
    sigma_w /= N_total

    sigma_b = np.zeros((D, D))
    for k in centroides:
        diff = (centroides[k] - centroide_global).reshape(-1, 1)
        sigma_b += diff @ diff.T
    sigma_b /= K

    try:
        rank_teorico = max(K - 1, 1)
        U, S, _ = np.linalg.svd(sigma_b)
        S_inv = np.zeros_like(S)
        S_inv[:rank_teorico] = 1.0 / S[:rank_teorico]
        sigma_b_pinv = (U[:, :rank_teorico] * S_inv[:rank_teorico]) @ U[:, :rank_teorico].T
        nc1 = float(np.trace(sigma_w @ sigma_b_pinv))
        nc1 = max(nc1, 0.0)
    except np.linalg.LinAlgError:
        nc1 = float('inf')
    return nc1


def calcular_nc1_concatenado(backbone, adapters, loader, num_classes_ativas):
    embeddings_por_classe = {k: [] for k in range(num_classes_ativas)}
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            phi = backbone(images)
            concat = torch.cat([ad(phi) for ad in adapters], dim=1).cpu().numpy()
            for feat, lbl in zip(concat, labels.numpy()):
                if lbl < num_classes_ativas:
                    embeddings_por_classe[lbl].append(feat)
    return _nc1_de_embeddings(embeddings_por_classe)


def calcular_matriz_scores_medios(backbone, adapters, fc, lvq, loader_teste, num_classes_ativas):
    soma_scores = {k: np.zeros(num_classes_ativas) for k in range(num_classes_ativas)}
    contagem = {k: 0 for k in range(num_classes_ativas)}
    with torch.no_grad():
        for images, labels in loader_teste:
            images = images.to(DEVICE)
            labels_np = labels.numpy()
            scores = calcular_logits_finais(backbone, adapters, fc, lvq, images, CLASSES_POR_TAREFA,
                                             ALPHA_LVQ, TAU_LVQ).cpu().numpy()
            for sc, lbl in zip(scores, labels_np):
                soma_scores[lbl] += sc
                contagem[lbl] += 1
    matriz = np.zeros((num_classes_ativas, num_classes_ativas))
    for i in range(num_classes_ativas):
        matriz[i] = soma_scores[i] / contagem[i] if contagem[i] > 0 else soma_scores[i]
    return matriz


def calcular_margem_vazamento(matriz, nomes_classes):
    n = matriz.shape[0]
    registros = []
    for c in range(n):
        proprio = matriz[c, c]
        concorrentes = {c2: matriz[c, c2] for c2 in range(n) if c2 != c}
        c_mais_forte = max(concorrentes, key=concorrentes.get)
        margem = float(proprio - concorrentes[c_mais_forte])
        registros.append({
            "classe": nomes_classes[c], "score_proprio": float(proprio),
            "concorrente_mais_forte": nomes_classes[c_mais_forte],
            "score_concorrente": float(concorrentes[c_mais_forte]), "margem_vazamento": margem,
        })
    return registros


# ============================================================================
# 14. GRÁFICOS (dimensionados dinamicamente para NUM_CLASSES_TOTAIS)
# ============================================================================
# Acima deste número de classes, os gráficos por classe passam a ser desenhados por bloco de tarefa
# (sem anotar célula por célula), para continuarem legíveis e com tamanho de arquivo razoável.
LIMIAR_CLASSES_DETALHADO = 20


def _mapa_cores(nome):
    try:
        import matplotlib
        return matplotlib.colormaps[nome]          # matplotlib >= 3.5
    except (AttributeError, KeyError):
        return cm.get_cmap(nome)                   # versões antigas (removido no 3.9)


def _paleta(num_classes):
    if num_classes <= 10:
        mapa, n = _mapa_cores('tab10'), 10
        return [mapa(i % n) for i in range(num_classes)]
    if num_classes <= 20:
        mapa, n = _mapa_cores('tab20'), 20
        return [mapa(i % n) for i in range(num_classes)]
    mapa = _mapa_cores('turbo')
    return [mapa(i / max(num_classes - 1, 1)) for i in range(num_classes)]


def _marcar_blocos_de_tarefa(ax, n):
    """Linhas nas fronteiras entre tarefas e rótulos 'T0', 'T1', ... no centro de cada bloco."""
    num_blocos = int(math.ceil(n / CLASSES_POR_TAREFA))
    for b in range(1, num_blocos):
        pos = b * CLASSES_POR_TAREFA - 0.5
        ax.axhline(pos, color='black', linewidth=0.8)
        ax.axvline(pos, color='black', linewidth=0.8)
    centros = [b * CLASSES_POR_TAREFA + (CLASSES_POR_TAREFA - 1) / 2 for b in range(num_blocos)]
    rotulos = [f"T{b}" for b in range(num_blocos)]
    ax.set_xticks(centros); ax.set_xticklabels(rotulos)
    ax.set_yticks(centros); ax.set_yticklabels(rotulos)


def obter_embeddings(backbone, adapter, loader):
    feats, labs = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            phi = backbone(images)
            emb = adapter(phi) if adapter is not None else phi
            feats.append(emb.cpu().numpy())
            labs.append(labels.numpy())
    return np.concatenate(feats), np.concatenate(labs)


def obter_embeddings_concatenados(backbone, adapters, loader):
    feats, labs = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            phi = backbone(images)
            concat = torch.cat([ad(phi) for ad in adapters], dim=1)
            feats.append(concat.cpu().numpy())
            labs.append(labels.numpy())
    return np.concatenate(feats), np.concatenate(labs)


def plotar_tsne(feats, labs, nomes_classe_cil, titulo, nome_arquivo):
    try:
        tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, max_iter=1000)
    except TypeError:
        tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, n_iter=1000)
    f2d = tsne.fit_transform(feats)
    presentes = np.unique(labs)
    plt.figure(figsize=(9, 7))
    if len(presentes) <= LIMIAR_CLASSES_DETALHADO:
        cores = _paleta(len(nomes_classe_cil))
        for c in presentes:
            idx = labs == c
            plt.scatter(f2d[idx, 0], f2d[idx, 1], c=[cores[int(c)]], label=nomes_classe_cil[int(c)],
                        alpha=0.6, s=25)
        plt.legend(fontsize=8, ncol=max(1, len(presentes) // 10))
    else:
        # muitas classes: uma cor por TAREFA (a pergunta relevante para vazamento é se as
        # tarefas se separam), pontos menores
        tarefas = labs // CLASSES_POR_TAREFA
        cores_t = _paleta(int(tarefas.max()) + 1)
        for t in np.unique(tarefas):
            idx = tarefas == t
            plt.scatter(f2d[idx, 0], f2d[idx, 1], c=[cores_t[int(t)]], label=f"Tarefa {int(t)}",
                        alpha=0.5, s=6)
        plt.legend(fontsize=9, markerscale=3)
    plt.title(titulo, fontsize=11, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    path = os.path.join(OS_FIGURES_DIR, f"{nome_arquivo}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f" [Gráfico] {path}")


def plotar_acc_por_classe(historico, classes_da_tarefa, nomes_classe_cil, indice_tarefa, nome_arquivo):
    epocas = [h["epoca"] for h in historico]
    cores = _paleta(len(classes_da_tarefa))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cor, c in zip(cores, classes_da_tarefa):
        nome = nomes_classe_cil[c]
        ax.plot(epocas, [h["acc_por_classe"][nome] for h in historico], label=nome, color=cor)
    ax.set_xlabel(f"Época de treino do Adapter_{indice_tarefa}")
    ax.set_ylabel("Acurácia de teste (%, score combinado cosseno+LVQ)")
    ax.set_ylim(-3, 103)
    ax.set_title(f"Acurácia por classe durante a Tarefa {indice_tarefa}", fontweight='bold', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=7 if len(classes_da_tarefa) > 10 else 9, ncol=2 if len(classes_da_tarefa) > 10 else 1)
    plt.tight_layout()
    path = os.path.join(OS_FIGURES_DIR, f"{nome_arquivo}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f" [Gráfico] {path}")


def plotar_aac_acumulado(historico, indice_tarefa, nome_arquivo):
    epocas = [h["epoca"] for h in historico]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(epocas, [h["aac"] for h in historico], color='tab:blue', label="AAC (tarefas vistas)")
    ax.set_xlabel(f"Época de treino do Adapter_{indice_tarefa}")
    ax.set_ylabel("AAC (%)")
    ax.set_ylim(-3, 103)
    ax.set_title(f"AAC acumulado durante a Tarefa {indice_tarefa}", fontweight='bold', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(OS_FIGURES_DIR, f"{nome_arquivo}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f" [Gráfico] {path}")


def plotar_matriz_confusao(y_true, y_pred, nomes_classe_cil, nome_arquivo):
    n = len(nomes_classe_cil)
    cm_ = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    nomes = [nomes_classe_cil[c] for c in range(n)]
    detalhado = n <= LIMIAR_CLASSES_DETALHADO
    tam = max(6.0, 1.1 * n) if detalhado else 12.0
    fig, ax = plt.subplots(figsize=(tam, tam))
    im = ax.imshow(cm_, cmap='Blues')
    ax.set_xlabel("Predito"); ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão — Nova Estratégia 2", fontweight='bold', fontsize=11)
    if detalhado:
        ax.set_xticks(range(n)); ax.set_xticklabels(nomes, rotation=45, ha='right', fontsize=max(6, 11 - n // 4))
        ax.set_yticks(range(n)); ax.set_yticklabels(nomes, fontsize=max(6, 11 - n // 4))
        valor_max = cm_.max() if cm_.max() > 0 else 1
        for i in range(n):
            for j in range(n):
                valor = cm_[i, j]
                cor = 'white' if valor > valor_max / 2 else 'black'
                ax.text(j, i, str(valor), ha='center', va='center', color=cor, fontweight='bold',
                         fontsize=max(5, 10 - n // 4))
    else:
        # blocos fora da diagonal principal de blocos = vazamento entre tarefas
        _marcar_blocos_de_tarefa(ax, n)
        ax.set_xlabel("Predito (classes agrupadas por tarefa)")
        ax.set_ylabel("Real (classes agrupadas por tarefa)")
    plt.colorbar(im, ax=ax, label='Nº de imagens')
    plt.tight_layout()
    path = os.path.join(OS_FIGURES_DIR, f"{nome_arquivo}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f" [Gráfico] {path}")


def plotar_mapa_calor_scores(matriz, nomes_classe_cil, nome_arquivo):
    n = len(nomes_classe_cil)
    nomes = [nomes_classe_cil[c] for c in range(n)]
    detalhado = n <= LIMIAR_CLASSES_DETALHADO
    tam = max(6.0, 1.1 * n) if detalhado else 12.0
    fig, ax = plt.subplots(figsize=(tam, tam))
    im = ax.imshow(matriz, cmap='RdYlGn', vmin=-1, vmax=1)
    ax.set_title("Mapa de Calor — Score Combinado Médio (cosseno+LVQ) por (Classe Real, Classe-Alvo)",
                 fontweight='bold', fontsize=9)
    if detalhado:
        ax.set_xticks(range(n)); ax.set_xticklabels([f"score_{x}" for x in nomes], rotation=45, ha='right',
                                                     fontsize=max(6, 11 - n // 4))
        ax.set_yticks(range(n)); ax.set_yticklabels([f"real {x}" for x in nomes], fontsize=max(6, 11 - n // 4))
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{matriz[i, j]:.2f}", ha='center', va='center', color='black', fontweight='bold',
                         fontsize=max(5, 9 - n // 4))
    else:
        _marcar_blocos_de_tarefa(ax, n)
        ax.set_xlabel("Classe-alvo do score (agrupadas por tarefa)")
        ax.set_ylabel("Classe real (agrupadas por tarefa)")
    plt.colorbar(im, ax=ax, label='score médio combinado')
    plt.tight_layout()
    path = os.path.join(OS_FIGURES_DIR, f"{nome_arquivo}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f" [Gráfico] {path}")


def plotar_margem_vazamento(registros, titulo, nome_arquivo):
    nomes = [r["classe"] for r in registros]
    valores = [r["margem_vazamento"] for r in registros]
    cores = ['tab:red' if v < 0 else 'tab:blue' for v in valores]
    detalhado = len(nomes) <= LIMIAR_CLASSES_DETALHADO
    fig, ax = plt.subplots(figsize=(max(9, 0.8 * len(nomes)) if detalhado else 18, 6))
    barras = ax.bar(range(len(nomes)), valores, color=cores)
    if detalhado:
        for barra, v in zip(barras, valores):
            ax.text(barra.get_x() + barra.get_width() / 2, v + (0.012 if v >= 0 else -0.02),
                     f"{v:+.3f}", ha='center', va='bottom' if v >= 0 else 'top', fontweight='bold', fontsize=8)
        ax.set_xticks(range(len(nomes))); ax.set_xticklabels(nomes, rotation=30, ha='right')
    else:
        # muitas classes: nomes pequenos na vertical, fronteiras de tarefa marcadas
        ax.set_xticks(range(len(nomes))); ax.set_xticklabels(nomes, rotation=90, fontsize=5)
        for b in range(1, int(math.ceil(len(nomes) / CLASSES_POR_TAREFA))):
            ax.axvline(b * CLASSES_POR_TAREFA - 0.5, color='gray', linestyle='--', linewidth=0.8)
        negativas = sum(v < 0 for v in valores)
        titulo = f"{titulo} — {negativas} de {len(valores)} classes com margem negativa"
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel("Margem de vazamento V_c = S̄(c,c) − S̄(c, concorrente mais forte)")
    ax.set_title(titulo, fontweight='bold', fontsize=11)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    path = os.path.join(OS_FIGURES_DIR, f"{nome_arquivo}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f" [Gráfico] {path}")


def plotar_drift_semantico(registros, nome_arquivo):
    rotulos = [f"{r['classe']}\n(após {r['momento']})" for r in registros]
    valores = [r['norma_delta'] for r in registros]
    detalhado = len(rotulos) <= LIMIAR_CLASSES_DETALHADO
    fig, ax = plt.subplots(figsize=(max(9, 0.8 * len(rotulos)) if detalhado else 18, 6))
    teto = max(valores + [1e-6])
    if detalhado:
        barras = ax.bar(rotulos, valores, color='tab:cyan')
        for barra, v in zip(barras, valores):
            ax.text(barra.get_x() + barra.get_width() / 2, v + teto * 0.03, f"{v:.2e}", ha='center',
                     fontweight='bold', fontsize=8)
    else:
        # muitas barras: uma cor por fronteira, sem rótulo por classe; o máximo vai no título
        momentos = sorted({r['momento'] for r in registros}, key=lambda m: int(m.split('_')[-1]))
        cores_m = _paleta(len(momentos))
        cores = [cores_m[momentos.index(r['momento'])] for r in registros]
        ax.bar(range(len(valores)), valores, color=cores)
        ax.set_xticks([])
        ax.set_xlabel("classes antigas, agrupadas pela fronteira em que o drift foi medido")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=cores_m[i], label=f"após {m}") for i, m in enumerate(momentos)],
                  fontsize=8)
        ax.text(0.01, 0.95, f"máximo = {teto:.2e}", transform=ax.transAxes, fontsize=9, va='top')
    ax.set_ylabel("‖Δ_c^(s→t)‖ — norma do drift semântico (Yu et al., 2020, Eq. 8)")
    ax.set_ylim(0, teto * 1.3 if teto > 0 else 1.0)
    ax.set_title("Drift semântico EXATO dos protótipos congelados, por fronteira de tarefa\n"
                 "(esperado ≈ 0: Adapter/backbone da classe antiga nunca recebem gradiente)",
                 fontweight='bold', fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    if detalhado:
        plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    path = os.path.join(OS_FIGURES_DIR, f"{nome_arquivo}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f" [Gráfico] {path}")


def plotar_diagnostico_ab(valores, nome_arquivo, rotulo_modulo):
    rotulos = [f"Original\n({rotulo_modulo} treinados)", "Teste A: Negação\n(θ_init − τ, ambos)",
               "Teste B: Exclusão\n(módulo fora da competição)",
               "Teste C: Aleatório\n(nunca treinado, não-degenerado)"]
    cores = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple']
    fig, ax = plt.subplots(figsize=(10.5, 6))
    barras = ax.bar(rotulos, valores, color=cores)
    for barra, v in zip(barras, valores):
        ax.text(barra.get_x() + barra.get_width() / 2, v + 1.5, f"{v:.1f}%", ha='center', fontweight='bold', fontsize=11)
    ax.axhline(50, color='gray', linestyle='--', alpha=0.6, label='referência')
    ax.set_ylabel("Acurácia das tarefas antigas (%)")
    ax.set_ylim(0, 108)
    ax.set_title(f"Diagnóstico A/B/C — Task Arithmetic (Ilharco et al. 2023), {rotulo_modulo}\n"
                 "Conteúdo negado (A) vs. Ausência (B) vs. Não-treinado genérico (C)",
                 fontweight='bold', fontsize=11)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    path = os.path.join(OS_FIGURES_DIR, f"{nome_arquivo}.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f" [Gráfico] {path}")


# ============================================================================
# 14b. [Estratégia 2] MÉTRICAS DE VAZAMENTO (metricas_vazamento.py)
# ============================================================================
def coletar_logits(backbone, adapters, fc, lvq, loader):
    todos_logits, todos_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = calcular_logits_finais(backbone, adapters, fc, lvq, images.to(DEVICE),
                                             CLASSES_POR_TAREFA, ALPHA_LVQ, TAU_LVQ)
            todos_logits.append(logits.cpu().numpy())
            todos_labels.append(labels.numpy())
    return np.concatenate(todos_logits), np.concatenate(todos_labels)


def estatisticas_score_tarefa(backbone, adapters, fc, lvq, loader_treino, indice_tarefa):
    """Média e desvio do score vencedor (entre as classes da própria tarefa) nas imagens de
    TREINO da tarefa, guardados ao fim dela. Usados para medir quanto do vazamento é só
    diferença de escala entre Adapters (kappa em metricas_vazamento.py)."""
    logits, _ = coletar_logits(backbone, adapters, fc, lvq, loader_treino)
    cols = list(range(indice_tarefa * CLASSES_POR_TAREFA, (indice_tarefa + 1) * CLASSES_POR_TAREFA))
    vencedor = logits[:, cols].max(axis=1)
    return float(vencedor.mean()), float(vencedor.std() + 1e-8)


def _json_seguro(v):
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else float(v)
    return v


def relatorio_vazamento(backbone, adapters, fc, lvq, loader, banco, task_stats, rotulo):
    logits, labels = coletar_logits(backbone, adapters, fc, lvq, loader)
    num_classes = logits.shape[1]
    class_task = np.arange(num_classes) // CLASSES_POR_TAREFA
    protos = np.stack([banco.media(c).cpu().numpy() for c in range(num_classes)])
    rep = leakage_metrics(logits, labels, class_task, protos=protos, task_stats=task_stats)
    logging.info(f"=== MÉTRICAS DE VAZAMENTO — {rotulo} ===")
    for linha in rep.resumo().splitlines():
        logging.info("   " + linha)
    return {
        "acc": _json_seguro(rep.acc), "vazamento": _json_seguro(rep.leakage),
        "vazamento_balanceado": _json_seguro(rep.leakage_bal), "erro_dentro_da_tarefa": _json_seguro(rep.e_wp),
        "parcela_do_erro_por_vazamento": _json_seguro(rep.leak_share),
        "acc_oraculo_tarefa": _json_seguro(rep.acc_oracle), "teto_de_ganho": _json_seguro(rep.gain_ceiling),
        "matriz_L": rep.L.tolist(), "contagens": rep.counts.tolist(),
        "indice_direcao": _json_seguro(rep.direction) if np.isfinite(rep.direction) else "inf",
        "perfil_deslocamento": {str(k): _json_seguro(v) for k, v in rep.offset_profile.items()},
        "spearman_recencia": _json_seguro(rep.rho_recency),
        "spearman_semantica": _json_seguro(rep.rho_semantic),
        "spearman_semantica_parcial": _json_seguro(rep.rho_semantic_partial),
        "vazamento_apos_normalizacao": _json_seguro(rep.leakage_norm),
        "parcela_atribuivel_a_escala": _json_seguro(rep.calib_share),
    }


# ============================================================================
# 15. PIPELINE PRINCIPAL
# ============================================================================
def executar_experimento():
    logging.info(f"=== [Nova Estratégia 2] Generalização da Estratégia 26 — "
                 f"{NUM_TASKS} tarefas x {CLASSES_POR_TAREFA} classes, dataset={CFG_DATASET['nome']}, "
                 f"seed={SEED} ===")

    train_set, test_set = carregar_dataset_base(CFG_DATASET)
    tarefas_classes_originais = montar_particao_tarefas(len(train_set.classes), NUM_TASKS, CLASSES_POR_TAREFA, SEED)
    mapa_original_para_cil, nomes_classe_cil = preparar_particao_e_nomes(train_set, tarefas_classes_originais)
    for t, tarefa_orig in enumerate(tarefas_classes_originais):
        nomes = [nomes_classe_cil[mapa_original_para_cil[c]] for c in tarefa_orig]
        logging.info(f"[Partição de tarefas] Tarefa {t}: classes originais {tarefa_orig} -> {nomes}")

    loaders_treino, loaders_teste_tarefa, loaders_teste_acumulado, loaders_teste_classe = obter_loaders_cil(
        train_set, test_set, tarefas_classes_originais, mapa_original_para_cil, BATCH_SIZE)

    backbone = criar_backbone_pretreinado_congelado()
    linhas, colunas = dimensionar_grade_lvq(NUM_TASKS, CLASSES_POR_TAREFA, UNIDADES_LVQ_POR_CLASSE)
    logging.info(f"[Mapa LVQ] Grade hexagonal {linhas}x{colunas}={linhas*colunas} unidades "
                 f"({NUM_TASKS} blocos de linha x {CLASSES_POR_TAREFA} blocos de coluna, "
                 f"{UNIDADES_LVQ_POR_CLASSE} unidades/classe).")
    lvq = criar_mapa_lvq(linhas, colunas, DIM_EMBEDDING, NUM_CLASSES_TOTAIS, NUM_TASKS, CLASSES_POR_TAREFA)

    json_data = {
        "estrategia": "nova_estrategia_2",
        "id_execucao": ID_EXECUCAO,              # [Estratégia 2] ex.: baseline_seed1993_20261106_101500
        "nome_config": NOME_CONFIG,              # [Estratégia 2] baseline / experimento
        "arquivo_config": CFG.get("_arquivo_config"),
        "configuracao_efetiva": {
            "seed": SEED, "dataset": CFG_DATASET["nome"], "num_tasks": NUM_TASKS,
            "classes_por_tarefa": CLASSES_POR_TAREFA, "batch_size": BATCH_SIZE,
            "epochs_por_tarefa": EPOCHS_POR_TAREFA, "embedding_dim": DIM_EMBEDDING,
            "r_adapter": R_ADAPTER, "lambda_lvq": LAMBDA_LVQ, "alpha_lvq": ALPHA_LVQ, "tau_lvq": TAU_LVQ,
            "som_linhas": linhas, "som_colunas": colunas, "otimizador": CFG_OTIMIZADOR,
        },
        "particao_tarefas": {
            "classes_originais_por_tarefa": tarefas_classes_originais,
            "nomes_por_tarefa": [[nomes_classe_cil[mapa_original_para_cil[c]] for c in tarefa]
                                 for tarefa in tarefas_classes_originais],
        },
        "tarefas": {},
        "diagnostico_task_arithmetic": {},
    }

    feats_antes, labs_antes = obter_embeddings(backbone, None, loaders_teste_acumulado[-1])
    plotar_tsne(feats_antes, labs_antes, nomes_classe_cil,
                "t-SNE — Todas as classes ANTES do treino (embedding cru do backbone)",
                "tsne_nova_estrategia_2_00_antes_do_treino")

    adapters = []
    fc = None
    prototipos_congelados = {}
    thetas_init_adapter = []
    thetas_init_lvq = []
    classes_por_fronteira = []
    checkpoints_adapters = []

    # [Estratégia 2] banco de gaussianas em phi + gerador de pseudo-negativos
    banco = BancoGaussianas(DIM_EMBEDDING, rank=int(CFG_PN.get("rank_cov", 32)),
                            encolhimento=float(CFG_PN.get("encolhimento", 1e-3)), device=DEVICE)
    gerador = GeradorPseudoNegativos(banco, CFG_PN, CLASSES_POR_TAREFA, ALPHA_LVQ, TAU_LVQ, DEVICE)
    task_stats = []
    json_data["configuracao_efetiva"]["pseudo_negativos"] = CFG_PN
    logging.info(f"[Estratégia 2] Pseudo-negativos: modo={PN_MODO} (\"nenhum\" reproduz a Nova Estratégia 1).")

    for t in range(NUM_TASKS):
        gerador.preparar_tarefa(classes_antigas=list(range(t * CLASSES_POR_TAREFA)))
        adapter_t, fc, historico_t, theta_init_adapter_t, theta_init_lvq_t, classes_novas = treinar_tarefa(
            t, backbone, lvq, adapters, fc, loaders_treino[t], loaders_teste_acumulado[t], nomes_classe_cil,
            gerador=gerador)

        # [Estratégia 2] alimenta o banco com phi das classes desta tarefa (backbone congelado:
        # não depende do Adapter, então estas estatísticas valem para todas as tarefas futuras)
        phi_por_classe = coletar_phi_por_classe(backbone, loaders_treino[t], classes_novas, DEVICE)
        for c, phi_c in phi_por_classe.items():
            banco.adicionar_classe(c, phi_c)
            logging.info(f"[Banco] {nomes_classe_cil[c]}: {phi_c.shape[0]} vetores phi -> gaussiana "
                         f"(posto {banco.U[c].shape[1]}, variância média por dimensão "
                         f"{banco.variancia_media(c):.4f}).")
        task_stats.append(estatisticas_score_tarefa(backbone, adapters + [adapter_t], fc, lvq,
                                                    loaders_treino[t], t))

        json_data["tarefas"][str(t)] = {
            "classes": [nomes_classe_cil[c] for c in classes_novas],
            "historico_treino": historico_t,
        }
        plotar_acc_por_classe(historico_t, classes_novas, nomes_classe_cil, t,
                              f"acc_por_classe_tarefa{t}_nova_estrategia_2")
        plotar_aac_acumulado(historico_t, t, f"aac_acumulado_tarefa{t}_nova_estrategia_2")

        for c in classes_novas:
            prototipos_congelados[c] = fc[c].clone()

        if t >= 1:
            feats_concat, labs_concat = obter_embeddings_concatenados(backbone, adapters + [adapter_t],
                                                                        loaders_teste_acumulado[t])
            plotar_tsne(feats_concat, labs_concat, nomes_classe_cil,
                        f"t-SNE — classes das tarefas 0..{t} (espaço conjunto dos Adapters)",
                        f"tsne_nova_estrategia_2_{t+1:02d}_pos_tarefa{t}")

            logging.info("=" * 60)
            logging.info(f"=== DIAGNÓSTICO A/B/C — fronteira tarefa {t-1} -> tarefa {t} ===")
            diagnostico = diagnostico_task_arithmetic_fronteira(
                t - 1, backbone, adapters, fc, lvq, adapter_t, theta_init_adapter_t, theta_init_lvq_t,
                classes_novas, loaders_teste_acumulado[t - 1], linhas, colunas, NUM_TASKS, CLASSES_POR_TAREFA)
            json_data["diagnostico_task_arithmetic"][f"{t-1}_para_{t}"] = diagnostico
            plotar_diagnostico_ab(
                [diagnostico["acc_original"], diagnostico["acc_negacao"],
                 diagnostico["acc_exclusao"], diagnostico["acc_aleatorio"]],
                f"diagnostico_task_arithmetic_nova_estrategia_2_f{t-1}_{t}",
                rotulo_modulo=f"Adapter_{t} + LVQ da tarefa {t}")

            logging.info(f"=== DRIFT SEMÂNTICO EXATO (Yu et al., 2020) — após a tarefa {t} ===")
            registros_drift_fronteira = []
            for tarefa_antiga in range(t):
                for c in range(tarefa_antiga * CLASSES_POR_TAREFA, (tarefa_antiga + 1) * CLASSES_POR_TAREFA):
                    norma, cos_sim, _ = calcular_drift_semantico(
                        backbone, adapters[tarefa_antiga], loaders_treino[tarefa_antiga], c,
                        prototipos_congelados[c], dim=DIM_EMBEDDING)
                    logging.info(f" > [Drift após tarefa {t}] {nomes_classe_cil[c]}: ||Δ||={norma:.8f} "
                                 f"(cos_sim={cos_sim:.8f})")
                    registros_drift_fronteira.append({
                        "classe": nomes_classe_cil[c], "momento": f"tarefa_{t}",
                        "norma_delta": norma, "cos_sim": cos_sim,
                    })
            json_data.setdefault("drift_semantico", []).extend(registros_drift_fronteira)

            num_classes_ate_aqui = (t + 1) * CLASSES_POR_TAREFA
            matriz_scores_fronteira = calcular_matriz_scores_medios(
                backbone, adapters + [adapter_t], fc, lvq, loaders_teste_acumulado[t], num_classes_ate_aqui)
            vazamento_fronteira = calcular_margem_vazamento(
                matriz_scores_fronteira, [nomes_classe_cil[c] for c in range(num_classes_ate_aqui)])
            for r in vazamento_fronteira:
                logging.info(f" > [Vazamento após tarefa {t}] {r['classe']}: margem={r['margem_vazamento']:+.4f} "
                             f"(próprio={r['score_proprio']:.4f}, concorrente mais forte="
                             f"{r['concorrente_mais_forte']}={r['score_concorrente']:.4f})")
            json_data.setdefault("vazamento_por_fronteira", {})[str(t)] = vazamento_fronteira
            plotar_margem_vazamento(vazamento_fronteira, f"Margem de vazamento após a tarefa {t}",
                                    f"margem_vazamento_nova_estrategia_2_apos_tarefa{t}")

            # [Estratégia 2] métricas de vazamento entre tarefas após esta fronteira
            json_data.setdefault("metricas_vazamento_por_fronteira", {})[str(t)] = relatorio_vazamento(
                backbone, adapters + [adapter_t], fc, lvq, loaders_teste_acumulado[t], banco,
                task_stats, f"após a tarefa {t}")

        adapters.append(adapter_t)
        thetas_init_adapter.append(theta_init_adapter_t)
        thetas_init_lvq.append(theta_init_lvq_t)
        classes_por_fronteira.append(classes_novas)

    # === RESULTADO FINAL ===
    logging.info("=" * 60)
    logging.info(f"=== RESULTADO FINAL ({NUM_TASKS} Adapters + LVQ, score combinado) ===")
    loader_teste_total = loaders_teste_acumulado[-1]
    labels_total, preds_total = prever_final(backbone, adapters, fc, lvq, loader_teste_total,
                                              CLASSES_POR_TAREFA, ALPHA_LVQ, TAU_LVQ)

    acc_por_tarefa_final = []
    for t in range(NUM_TASKS):
        labels_t, preds_t = prever_final(backbone, adapters, fc, lvq, loaders_teste_tarefa[t],
                                          CLASSES_POR_TAREFA, ALPHA_LVQ, TAU_LVQ)
        acc_t = float((preds_t == labels_t).mean() * 100)
        acc_por_tarefa_final.append(acc_t)
        logging.info(f" > Acc tarefa {t} ({', '.join(nomes_classe_cil[c] for c in classes_por_fronteira[t])}): "
                     f"{acc_t:.2f}%")
    aac = float(np.mean(acc_por_tarefa_final))
    logging.info(f" > AAC ({NUM_TASKS} tarefas): {aac:.2f}%")

    nc1 = calcular_nc1_concatenado(backbone, adapters, loader_teste_total, NUM_CLASSES_TOTAIS)
    matriz_scores = calcular_matriz_scores_medios(backbone, adapters, fc, lvq, loader_teste_total, NUM_CLASSES_TOTAIS)
    alinhamento_diagonal = {nomes_classe_cil[c]: float(matriz_scores[c, c]) for c in range(NUM_CLASSES_TOTAIS)}

    json_data["resultado_final"] = {
        "acc_por_tarefa": acc_por_tarefa_final, "aac": aac, "nc1_concatenado": nc1,
        "alinhamento_diagonal": alinhamento_diagonal, "matriz_scores_medios": matriz_scores.tolist(),
    }

    plotar_matriz_confusao(labels_total, preds_total, nomes_classe_cil, "matriz_confusao_nova_estrategia_2")
    plotar_mapa_calor_scores(matriz_scores, nomes_classe_cil, "mapa_calor_similaridade_nova_estrategia_2")

    vazamento_final = calcular_margem_vazamento(matriz_scores, [nomes_classe_cil[c] for c in range(NUM_CLASSES_TOTAIS)])
    for r in vazamento_final:
        logging.info(f" > [Vazamento final] {r['classe']}: margem={r['margem_vazamento']:+.4f} "
                     f"(próprio={r['score_proprio']:.4f}, concorrente mais forte={r['concorrente_mais_forte']}"
                     f"={r['score_concorrente']:.4f})")
    json_data["vazamento_por_classe_final"] = vazamento_final
    plotar_margem_vazamento(vazamento_final, "Margem de vazamento final (todas as classes)",
                            "margem_vazamento_nova_estrategia_2_final")

    if "drift_semantico" in json_data:
        plotar_drift_semantico(json_data["drift_semantico"], "drift_semantico_nova_estrategia_2")

    # [Estratégia 2] métricas de vazamento finais
    json_data["metricas_vazamento_final"] = relatorio_vazamento(
        backbone, adapters, fc, lvq, loader_teste_total, banco, task_stats, "final")

    # === CHECKPOINT ===
    checkpoint_path = os.path.join(OS_CHECKPOINTS_DIR, f"checkpoint_{ID_EXECUCAO}.pt")
    torch.save({
        "adapters_state_dicts": [ad.state_dict() for ad in adapters],
        "thetas_init_adapter": thetas_init_adapter,
        "thetas_init_lvq": thetas_init_lvq,
        "lvq_state_dict": lvq.state_dict(),
        "fc": fc.cpu(),
        "prototipos_congelados": {k: v.cpu() for k, v in prototipos_congelados.items()},
        "tarefas_classes_originais": tarefas_classes_originais,
        "mapa_original_para_cil": mapa_original_para_cil,
        "nomes_classe_cil": nomes_classe_cil,
        "r_adapter": R_ADAPTER, "dim_embedding": DIM_EMBEDDING,
        "seed": SEED, "configuracao": CFG,
        "banco_gaussianas": banco.estado(),          # [Estratégia 2]
        "estatisticas_score_por_tarefa": task_stats,  # [Estratégia 2]
    }, checkpoint_path)
    logging.info(f" [Checkpoint] {checkpoint_path}")

    json_filename = os.path.join(OS_RESULTS_DIR, f"results_{ID_EXECUCAO}.json")
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)
    logging.info(f" [JSON] {json_filename}")


if __name__ == "__main__":
    executar_experimento()
