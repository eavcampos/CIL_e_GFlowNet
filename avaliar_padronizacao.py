#!/usr/bin/env python3
"""
Avalia, SEM TREINAR DE NOVO, o efeito de padronizar as pontuações de cada Adapter (especialista)
antes de escolher a classe, usando os checkpoints já salvos em resultados/checkpoints/.

Padronização (a mesma do kappa em metricas_vazamento.py):
    nota_padronizada(c) = (nota(c) - media_k) / desvio_k,   k = tarefa da classe c
media_k e desvio_k são a média e o desvio da nota vencedora nas imagens de TREINO da tarefa k,
calculados ao fim da tarefa k e guardados no checkpoint ("estatisticas_score_por_tarefa").
O teste não é usado para obtê-los, então a avaliação é honesta.

Também testa uma variante mais simples, "centrar": só subtrai a média (sem dividir pelo desvio).

Como a padronização é a mesma transformação crescente para todas as classes de uma tarefa, ela
NÃO muda qual classe vence dentro de cada tarefa: só muda QUAL TAREFA vence. Ou seja, mexe só no
vazamento; o erro dentro da tarefa (com o oráculo) fica igual.

Uso (na pasta do projeto, com o mesmo Python das execuções):
    python avaliar_padronizacao.py
    python avaliar_padronizacao.py --checkpoints resultados/checkpoints --saida resultados/padronizacao
    python avaliar_padronizacao.py --perfil imagenet_r     # execuções feitas com perfis/imagenet_r.json

Saída: resultados/padronizacao/<data_hora>/
    por_execucao.csv  -> uma linha por (config, seed, versão das notas)
    resumo.csv        -> média e desvio por braço e versão, e diferenças pareadas
e um resumo no terminal. Leva poucos minutos por checkpoint (só avaliação no teste).
"""
import argparse
import csv
import json
import glob
import os
import re
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

PASTA = os.path.dirname(os.path.abspath(__file__))
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ORDEM = ["ncm", "baseline", "gauss", "experimento", "margem_padronizada"]


class Adapter(nn.Module):
    """Igual ao de cil_nova_estrategia_2.py."""
    def __init__(self, dim, r, residual):
        super().__init__()
        self.down = nn.Linear(dim, r)
        self.up = nn.Linear(r, dim)
        self.residual = residual

    def forward(self, x):
        saida = self.up(F.relu(self.down(x)))
        return x + saida if self.residual else saida


def backbone_congelado():
    try:
        rede = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    except AttributeError:
        rede = torchvision.models.resnet18(pretrained=True)
    rede.fc = nn.Identity()
    for p in rede.parameters():
        p.requires_grad = False
    return rede.eval().to(DEVICE)


def resolver(caminho):
    caminho = os.path.expanduser(caminho)
    return caminho if os.path.isabs(caminho) else os.path.normpath(os.path.join(PASTA, caminho))


_CACHE_TESTE = {}


def construir_transform(cfg_dataset):
    """Igual a cil_nova_estrategia_2.py."""
    resize = tuple(cfg_dataset.get("resize", [64, 64]))
    if cfg_dataset.get("recorte_central", False):
        passos = [transforms.Resize(int(round(max(resize) / 0.875))), transforms.CenterCrop(resize)]
    else:
        passos = [transforms.Resize(resize)]
    return transforms.Compose(passos + [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def conjunto_teste(cfg_dataset):
    """Conjunto de teste com os rótulos ORIGINAIS do dataset (o checkpoint traz o mapa para o
    espaço CIL). Reproduz a lógica de carregar_dataset_base de cil_nova_estrategia_2.py."""
    nome = cfg_dataset["nome"].strip().lower()
    chave = json.dumps(cfg_dataset, sort_keys=True)
    if chave in _CACHE_TESTE:
        return _CACHE_TESTE[chave]
    tf = construir_transform(cfg_dataset)
    if nome in ("cifar10", "cifar100"):
        raiz = resolver(cfg_dataset.get("raiz", cfg_dataset.get("raiz_cifar", "../data")))
        classe = torchvision.datasets.CIFAR10 if nome == "cifar10" else torchvision.datasets.CIFAR100
        ds = classe(root=raiz, train=False, download=False, transform=tf)
    else:
        padrao = cfg_dataset.get("raiz_omnibenchmark") if nome == "omnibenchmark" else None
        raiz = resolver(cfg_dataset.get("raiz", padrao or f"../../my_data/{nome}"))
        if cfg_dataset.get("pasta_unica") is not None:
            base = torchvision.datasets.ImageFolder(os.path.join(raiz, cfg_dataset["pasta_unica"]), transform=tf)
            rng = np.random.RandomState(int(cfg_dataset.get("seed_divisao", 0)))
            fr = float(cfg_dataset.get("fracao_teste", 0.2))
            por_classe, teste = {}, []
            for a in base.samples:
                por_classe.setdefault(a[1], []).append(a)
            for r in sorted(por_classe):
                lista = por_classe[r]
                ordem = rng.permutation(len(lista))
                n_teste = max(1, int(round(fr * len(lista))))
                teste += [lista[i] for i in ordem[:n_teste]]
            base.samples = base.imgs = teste
            base.targets = [r for _, r in teste]
            ds = base
        else:
            nomes_treino = torchvision.datasets.ImageFolder(
                os.path.join(raiz, cfg_dataset.get("pasta_treino", "train"))).classes
            ds = torchvision.datasets.ImageFolder(os.path.join(raiz, cfg_dataset.get("pasta_teste", "test")),
                                                  transform=tf)
            idx = {n: i for i, n in enumerate(nomes_treino)}
            ds.samples = [(c, idx[ds.classes[r]]) for c, r in ds.samples]
            ds.imgs, ds.targets = ds.samples, [r for _, r in ds.samples]
    _CACHE_TESTE[chave] = ds
    return ds


@torch.no_grad()
def phi_do_teste(backbone, ds):
    """Descrições phi de todo o teste (o backbone é o mesmo em todas as execuções: calcula uma vez)."""
    chave = ("phi", id(ds))
    if chave in _CACHE_TESTE:
        return _CACHE_TESTE[chave]
    feats, rotulos = [], []
    for x, y in DataLoader(ds, batch_size=256, shuffle=False, num_workers=2):
        feats.append(backbone(x.to(DEVICE)).cpu())
        rotulos.append(y)
    res = (torch.cat(feats), torch.cat(rotulos).numpy())
    _CACHE_TESTE[chave] = res
    return res


@torch.no_grad()
def notas(ck, phi):
    """Notas (cosseno) de todas as classes, como em calcular_logits_finais com alpha_lvq = 0."""
    cfg = ck["configuracao"]
    treino = cfg["treino"]
    alpha = float(treino.get("alpha_lvq", 0.4))
    if alpha != 0:
        raise SystemExit("Este script só trata alpha_lvq = 0 (LVQ desligado), como nos configs do estudo.")
    cpt = int(cfg["tarefas"]["classes_por_tarefa"])
    residual = bool(treino.get("adapter_residual", False))
    adapters = []
    for sd in ck["adapters_state_dicts"]:
        ad = Adapter(int(ck["dim_embedding"]), int(ck["r_adapter"]), residual)
        ad.load_state_dict(sd)
        adapters.append(ad.eval().to(DEVICE))
    fc = ck["fc"].to(DEVICE)
    fc = fc / fc.norm(dim=1, keepdim=True).clamp_min(1e-12)
    saida = []
    for i in range(0, len(phi), 1024):
        p = phi[i:i + 1024].to(DEVICE)
        cols = []
        for k, ad in enumerate(adapters):
            e = F.normalize(ad(p), dim=1)
            cols.append(e @ fc[k * cpt:(k + 1) * cpt].T)
        saida.append(torch.cat(cols, dim=1).cpu())
    return torch.cat(saida).numpy(), cpt


def padronizar(S, cpt, stats, modo):
    Z = S.astype(float).copy()
    for k, (mu, sd) in enumerate(stats):
        cols = slice(k * cpt, (k + 1) * cpt)
        Z[:, cols] = (Z[:, cols] - mu) / (max(sd, 1e-8) if modo == "zscore" else 1.0)
    return Z


def metricas(S, y, cpt, n_tarefas):
    """Acurácia final, vazamento, erro dentro da tarefa, oráculo, acurácia por tarefa e AAC
    incremental. As notas de uma classe só dependem do Adapter e do protótipo da tarefa dela, que
    ficam congelados ao fim da tarefa; então a avaliação 'após a tarefa t' é a mesma matriz
    restrita às classes e às imagens das tarefas 0..t."""
    tarefa = lambda c: c // cpt
    pred = S.argmax(1)
    tau, t_hat = tarefa(y), tarefa(pred)
    acc = (pred == y).mean()
    vaz = (t_hat != tau).mean()
    ewp = ((t_hat == tau) & (pred != y)).mean()
    oraculo = np.mean([S[i, tau[i] * cpt:(tau[i] + 1) * cpt].argmax() + tau[i] * cpt == y[i] for i in range(len(y))])
    por_tarefa = [float((pred[tau == k] == y[tau == k]).mean() * 100) for k in range(n_tarefas)]
    apos = []
    for t in range(n_tarefas):
        m = y < (t + 1) * cpt
        apos.append(float((S[m][:, :(t + 1) * cpt].argmax(1) == y[m]).mean() * 100))
    return {"acc_final": 100 * acc, "aac_incremental": float(np.mean(apos)), "vazamento": 100 * vaz,
            "erro_dentro_da_tarefa": 100 * ewp, "acc_oraculo": 100 * oraculo, "acc_por_tarefa": por_tarefa,
            "acc_apos_tarefa": apos}


def papel(nome_config):
    """'gauss_imagenet_r' -> ('gauss', '_imagenet_r'); 'gauss' -> ('gauss', '')."""
    for b in ORDEM:
        if nome_config == b or nome_config.startswith(b + "_"):
            return b, nome_config[len(b):]
    return None, None


def mais_recentes(pasta):
    """{(config, seed): caminho} com o checkpoint mais recente de cada combinação."""
    achados = {}
    for f in glob.glob(os.path.join(pasta, "checkpoint_*.pt")):
        m = re.match(r"checkpoint_(.+)_seed(\d+)_(\d{8}_\d{6})\.pt$", os.path.basename(f))
        if not m:
            continue
        chave = (m.group(1), int(m.group(2)))
        if chave not in achados or m.group(3) > achados[chave][0]:
            achados[chave] = (m.group(3), f)
    return {k: v[1] for k, v in achados.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoints", default=os.path.join(PASTA, "resultados", "checkpoints"))
    ap.add_argument("--saida", default=os.path.join(PASTA, "resultados", "padronizacao"))
    ap.add_argument("--perfil", dest="grupo", default="",
                    help="avalia as execuções feitas com esse perfil de dataset (ex.: imagenet_r); "
                         "vazio = execuções sem perfil (CIFAR-100 dos configs)")
    args = ap.parse_args()

    ap_grupo = args.grupo
    ckpts = {}
    for (cfg_nome, seed), caminho in mais_recentes(args.checkpoints).items():
        braco, grupo = papel(cfg_nome)
        if braco is not None and grupo.strip("_") == ap_grupo:
            ckpts[(braco, seed)] = caminho
    if not ckpts:
        raise SystemExit(f"Nenhum checkpoint de {ORDEM}{' com perfil ' + ap_grupo if ap_grupo else ''} em "
                         f"{os.path.abspath(args.checkpoints)}")
    print(f"Dispositivo: {DEVICE}. {len(ckpts)} checkpoints: "
          + ", ".join(f"{c}/{s}" for c, s in sorted(ckpts, key=lambda k: (ORDEM.index(k[0]), k[1]))))

    backbone = backbone_congelado()
    linhas = []
    for (config, seed) in sorted(ckpts, key=lambda k: (ORDEM.index(k[0]), k[1])):
        try:
            ck = torch.load(ckpts[(config, seed)], map_location="cpu", weights_only=False)
        except TypeError:
            ck = torch.load(ckpts[(config, seed)], map_location="cpu")
        ds = conjunto_teste(ck["configuracao"]["dataset"])
        phi, y_orig = phi_do_teste(backbone, ds)
        mapa = ck["mapa_original_para_cil"]
        manter = np.array([int(v) in mapa for v in y_orig])   # classes fora das tarefas (num_tasks parcial)
        y = np.array([mapa[int(v)] for v in y_orig[manter]])
        S, cpt = notas(ck, phi[torch.from_numpy(manter)])
        n_tarefas = S.shape[1] // cpt
        stats = ck["estatisticas_score_por_tarefa"]
        for versao, matriz in [("original", S), ("padronizada", padronizar(S, cpt, stats, "zscore")),
                               ("centrada", padronizar(S, cpt, stats, "centrar"))]:
            m = metricas(matriz, y, cpt, n_tarefas)
            linhas.append({"config": config, "seed": seed, "versao": versao, **m})
        o, p = linhas[-3], linhas[-2]
        print(f"  {config:<11} {seed}: acurácia final {o['acc_final']:6.2f} -> {p['acc_final']:6.2f} | "
              f"vazamento {o['vazamento']:6.2f} -> {p['vazamento']:6.2f} | "
              f"AAC incremental {o['aac_incremental']:6.2f} -> {p['aac_incremental']:6.2f}")

    pasta = os.path.join(args.saida, f"{datetime.now():%Y%m%d_%H%M%S}" + (f"_{ap_grupo}" if ap_grupo else ""))
    os.makedirs(pasta, exist_ok=True)
    chaves = ["acc_final", "aac_incremental", "vazamento", "erro_dentro_da_tarefa", "acc_oraculo"]
    n_t = len(linhas[0]["acc_por_tarefa"])
    with open(os.path.join(pasta, "por_execucao.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["config", "seed", "versao"] + chaves + [f"acc_T{t}" for t in range(n_t)])
        for l in linhas:
            w.writerow([l["config"], l["seed"], l["versao"]] + [f"{l[k]:.3f}" for k in chaves]
                       + [f"{v:.3f}" for v in l["acc_por_tarefa"]])

    tab = {}
    for l in linhas:
        tab.setdefault((l["config"], l["versao"]), {})[l["seed"]] = l
    seeds = sorted({l["seed"] for l in linhas})
    print("\nMÉDIA ± DESVIO ENTRE SEEDS")
    with open(os.path.join(pasta, "resumo.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["config", "versao", "n_seeds"] + [f"{k}_media" for k in chaves] + [f"{k}_desvio" for k in chaves]
                   + [f"acc_T{t}_media" for t in range(n_t)])
        for config in ORDEM:
            for versao in ["original", "padronizada", "centrada"]:
                g = tab.get((config, versao))
                if not g:
                    continue
                vals = {k: [g[s][k] for s in g] for k in chaves}
                med = {k: np.mean(v) for k, v in vals.items()}
                dp = {k: (np.std(v, ddof=1) if len(v) > 1 else 0.0) for k, v in vals.items()}
                pt = np.mean([g[s]["acc_por_tarefa"] for s in g], axis=0)
                w.writerow([config, versao, len(g)] + [f"{med[k]:.3f}" for k in chaves]
                           + [f"{dp[k]:.3f}" for k in chaves] + [f"{v:.3f}" for v in pt])
                print(f"  {config:<11} {versao:<11} acc final {med['acc_final']:6.2f}±{dp['acc_final']:4.2f} | "
                      f"AAC inc {med['aac_incremental']:6.2f}±{dp['aac_incremental']:4.2f} | "
                      f"vazamento {med['vazamento']:6.2f} | erro dentro {med['erro_dentro_da_tarefa']:5.2f} | "
                      f"por tarefa {np.round(pt, 1).tolist()}")

        print("\nDIFERENÇAS PAREADAS NA ACURÁCIA FINAL (média; seeds a favor)")
        w.writerow([])
        w.writerow(["comparacao", "metrica", "diferenca_media", "seeds_a_favor", "diferencas_por_seed"])
        comps = [(("baseline", "padronizada"), ("baseline", "original")),
                 (("gauss", "padronizada"), ("gauss", "original")),
                 (("experimento", "padronizada"), ("experimento", "original")),
                 (("gauss", "padronizada"), ("ncm", "original")),
                 (("experimento", "padronizada"), ("ncm", "original")),
                 (("experimento", "padronizada"), ("gauss", "padronizada")),
                 (("gauss", "padronizada"), ("baseline", "padronizada"))]
        for a, b in comps:
            if a not in tab or b not in tab:
                continue
            comum = [s for s in seeds if s in tab[a] and s in tab[b]]
            for k in ["acc_final", "aac_incremental"]:
                d = np.array([tab[a][s][k] - tab[b][s][k] for s in comum])
                nome = f"{a[0]}({a[1]}) - {b[0]}({b[1]})"
                w.writerow([nome, k, f"{d.mean():+.3f}", f"{int((d > 0).sum())}/{len(d)}",
                            " ".join(f"{v:+.2f}" for v in d)])
                print(f"  {nome:<46} {k:<15} {d.mean():+6.2f}  ({int((d > 0).sum())}/{len(d)} seeds)  "
                      f"{np.round(d, 2).tolist()}")
    print(f"\nArquivos em {os.path.abspath(pasta)}")


if __name__ == "__main__":
    main()
