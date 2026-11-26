"""
[Fase B, parte 2] Braços SEM adapters por tarefa do plano de análise (Seções 4.3.2 e 4.3.3):

  fecam    : FeCAM sem treino (Goswami et al., NeurIPS 2023). Distância de Mahalanobis com a covariância
             completa de cada classe, com encolhimento e normalização para correlação, como no repositório
             oficial (dipamgoswami/FeCAM, conferido em 22/11/2026).
  gauss_ca : classificador de cosseno único sobre phi, realinhado depois de cada tarefa com features
             sorteadas do mesmo banco de gaussianas dos nossos braços (etapa de "alinhamento do
             classificador" do SLCA, Zhang et al., ICCV 2023).

Não é executado diretamente: o cil_nova_estrategia_2.py chama executar(base) quando o config tem
treino.classificador = "fecam" ou "gauss_ca". Assim, os dois braços usam exatamente o mesmo carregamento
de dados, a mesma partição de classes, o mesmo cache de phi, as mesmas métricas de vazamento e o mesmo
formato de JSON dos outros braços.

Uso (como os demais configs):
    python cil_nova_estrategia_2.py --config config_fecam.json --perfil perfis/cenario5_imagenet_r_inc20_vit.json --seed 1997
"""
import os
import json
import time
import logging

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------------------------------------
# FeCAM
# ---------------------------------------------------------------------------------------------------------
def hiperparametros_fecam(base):
    """Valores do repositório oficial do FeCAM (plano de análise, Seção 4.3.2). Podem ser sobrescritos por
    um bloco "fecam" no config: {"alpha1": ..., "alpha2": ..., "tukey_beta": ...}."""
    if base.BACKBONE == "resnet18":
        hp = {"alpha1": 1.0, "alpha2": 1.0, "tukey_beta": 0.5}          # exps/FeCAM_cifar100.json
    elif base.CFG_DATASET["nome"].lower() in ("cifar100", "cifar10"):
        hp = {"alpha1": 1.0, "alpha2": 0.0, "tukey_beta": None}         # FeCAM_vit_cifar100.py
    else:
        hp = {"alpha1": 10.0, "alpha2": 10.0, "tukey_beta": None}       # FeCAM_vit_imagenet-r.py
    hp.update(base.CFG.get("fecam", {}) or {})
    return hp


def _transformar_fecam(phi, tukey_beta):
    """Como no FeCAM oficial: normaliza cada vetor (norma 1); com a ResNet, aplica depois a transformação de
    Tukey (potência beta), que só faz sentido para valores não negativos."""
    x = F.normalize(phi.double(), dim=1, eps=1e-8)
    if tukey_beta is not None:
        x = x.clamp(min=0).pow(tukey_beta)
    return x


def _encolher_e_normalizar(cov, alpha1, alpha2):
    """shrink_cov + normalize_cov do FeCAM oficial: soma alpha1 * (média da diagonal) à diagonal e
    alpha2 * (média fora da diagonal) fora dela; depois converte para matriz de correlação."""
    d = cov.shape[0]
    iden = torch.eye(d, dtype=cov.dtype, device=cov.device)
    fora = cov * (1 - iden)
    mascara = fora != 0
    media_fora = fora[mascara].mean() if mascara.any() else torch.zeros((), dtype=cov.dtype, device=cov.device)
    cov = cov + alpha1 * torch.diagonal(cov).mean() * iden + alpha2 * media_fora * (1 - iden)
    sd = torch.sqrt(torch.diagonal(cov).clamp(min=1e-12))
    return cov / (sd[:, None] * sd[None, :])


class ClassificadorFeCAM:
    def __init__(self, hp, device):
        self.hp, self.device = hp, device
        self.medias, self.inversas = {}, {}

    def adicionar_classe(self, c, phi_c):
        x = _transformar_fecam(phi_c.to(self.device), self.hp["tukey_beta"])
        mu = x.mean(0)
        cov = torch.cov(x.T) if x.shape[0] > 1 else torch.zeros(x.shape[1], x.shape[1], dtype=x.dtype,
                                                                 device=x.device)
        cov = _encolher_e_normalizar(cov, self.hp["alpha1"], self.hp["alpha2"])
        self.medias[c] = mu
        self.inversas[c] = torch.linalg.pinv(cov).float().cpu()   # guardadas na CPU (d x d por classe)

    @torch.no_grad()
    def notas(self, phi, classes):
        """Notas = -distância de Mahalanobis (maior = mais parecido), shape [n, len(classes)]."""
        x = _transformar_fecam(phi.to(self.device), self.hp["tukey_beta"])
        usar_norma = self.hp["tukey_beta"] is not None   # ResNet: o FeCAM renormaliza depois de Tukey
        if usar_norma:
            x = F.normalize(x, dim=1, eps=1e-8)
        x = x.float()
        cols = []
        for c in classes:
            mu = self.medias[c].float()
            if usar_norma:
                mu = F.normalize(mu, dim=0, eps=1e-8)
            dif = x - mu
            A = self.inversas[c].to(self.device)
            cols.append(-((dif @ A) * dif).sum(1))
        return torch.stack(cols, dim=1)

    def bytes_por_classe(self, d):
        return 4 * (d + d * d)   # média + covariância completa


# ---------------------------------------------------------------------------------------------------------
# gauss_ca
# ---------------------------------------------------------------------------------------------------------
class ClassificadorGaussCA:
    """Um vetor por classe; nota = cosseno(phi, vetor). Depois de cada tarefa, os vetores de TODAS as classes
    vistas são realinhados com features sorteadas do banco (plano, Seção 4.3.3: 5 épocas, 256 amostras por
    classe vista por época, SGD 0,01 com momentum 0,9, lote 48, escala s = 16)."""

    def __init__(self, base, device):
        self.base, self.device = base, device
        self.W = torch.zeros(0, base.DIM_EMBEDDING, device=device)
        cfg = base.CFG.get("gauss_ca", {}) or {}
        self.epocas = int(cfg.get("epocas", 5))
        self.n_por_classe = int(cfg.get("amostras_por_classe", 256))
        self.lr = float(cfg.get("lr", 0.01))
        self.momentum = float(cfg.get("momentum", 0.9))
        self.lote = int(cfg.get("lote", base.BATCH_SIZE))
        self.escala = float(cfg.get("escala", base.ESCALA_LOGITS))
        self.relu = bool(base.CFG_PN.get("relu", True))

    def adicionar_classes(self, medias_phi):
        """medias_phi: tensor [n_novas, d], na ordem das classes novas. Vetores iniciados nas médias de phi."""
        self.W = torch.cat([self.W, medias_phi.to(self.device).float()], dim=0)

    def realinhar(self, banco, classes_vistas):
        W = torch.nn.Parameter(self.W.clone())
        opt = torch.optim.SGD([W], lr=self.lr, momentum=self.momentum)
        rot_base = torch.tensor(classes_vistas, device=self.device).repeat_interleave(self.n_por_classe)
        perdas = []
        for _ in range(self.epocas):
            with torch.no_grad():
                phi = banco.amostrar(rot_base)
                if self.relu:
                    phi = F.relu(phi)
                ordem = torch.randperm(len(rot_base), device=self.device)
            total, n = 0.0, 0
            for i in range(0, len(ordem), self.lote):
                idx = ordem[i:i + self.lote]
                logits = F.normalize(phi[idx], dim=1, eps=1e-8) @ F.normalize(W, dim=1, eps=1e-8).T
                perda = F.cross_entropy(self.escala * logits, rot_base[idx])
                opt.zero_grad()
                perda.backward()
                opt.step()
                total += perda.item() * len(idx)
                n += len(idx)
            perdas.append(total / max(n, 1))
        self.W = W.detach()
        return perdas

    @torch.no_grad()
    def notas(self, phi, classes):
        return F.normalize(phi.to(self.device), dim=1, eps=1e-8) @ F.normalize(self.W[classes], dim=1, eps=1e-8).T


# ---------------------------------------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------------------------------------
def _avaliar(clf, phi_teste, y_teste, classes, C):
    notas = clf.notas(phi_teste, classes).cpu().numpy()
    return notas, y_teste.cpu().numpy()


def executar(base):
    """Mesmo esqueleto de executar_experimento() do script principal, sem adapters nem LVQ."""
    B = base
    t_inicio = time.time()
    tipo = B.CLASSIFICADOR
    C = B.CLASSES_POR_TAREFA
    if not B.CACHE_PHI:
        raise ValueError("Os braços fecam e gauss_ca trabalham sobre phi em cache: use treino.cache_phi = true.")
    logging.info(f"=== [Fase B] Braço sem adapters: {tipo} — dataset={B.CFG_DATASET['nome']}, seed={B.SEED} ===")

    train_set, test_set = B.carregar_dataset_base(B.CFG_DATASET)
    num_tasks = B.NUM_TASKS or len(train_set.classes) // C
    B.NUM_TASKS, B.NUM_CLASSES_TOTAIS = num_tasks, num_tasks * C
    tarefas_orig = B.montar_particao_tarefas(len(train_set.classes), num_tasks, C, B.SEED, ordem=B.ORDEM_CLASSES)
    mapa, nomes = B.preparar_particao_e_nomes(train_set, tarefas_orig)

    backbone_real = B.criar_backbone_pretreinado_congelado()
    t_cache = time.time()
    phi_tr, phi_te = B.preparar_cache_phi(backbone_real, train_set, test_set, B.CFG_DATASET)
    tempo_cache = time.time() - t_cache
    if phi_tr.shape[1] != B.DIM_EMBEDDING:
        raise ValueError(f"phi tem dimensão {phi_tr.shape[1]}, mas treino.embedding_dim = {B.DIM_EMBEDDING}.")
    del backbone_real
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    train_set = B.DatasetPhi(phi_tr.to(B.DEVICE), train_set.targets, train_set.classes)
    test_set = B.DatasetPhi(phi_te.to(B.DEVICE), test_set.targets, test_set.classes)
    loaders_treino, loaders_teste_tarefa, loaders_teste_acum, _, _ = B.obter_loaders_cil(
        train_set, test_set, tarefas_orig, mapa, B.BATCH_SIZE, fracao_validacao=0.0, seed=B.SEED)

    banco = B.BancoGaussianas(B.DIM_EMBEDDING, rank=int(B.CFG_PN.get("rank_cov", 32)),
                              encolhimento=float(B.CFG_PN.get("encolhimento", 1e-3)), device=B.DEVICE)
    if tipo == "fecam":
        hp = hiperparametros_fecam(B)
        clf = ClassificadorFeCAM(hp, B.DEVICE)
        logging.info(f"[FeCAM] hiperparâmetros: {hp}")
    else:
        hp = None
        clf = ClassificadorGaussCA(B, B.DEVICE)
        logging.info(f"[gauss_ca] {clf.epocas} épocas por tarefa, {clf.n_por_classe} amostras por classe vista, "
                     f"lr {clf.lr}, momentum {clf.momentum}, lote {clf.lote}, escala {clf.escala}, ReLU {clf.relu}")

    json_data = {
        "estrategia": "nova_estrategia_2", "id_execucao": B.ID_EXECUCAO, "nome_config": B.NOME_CONFIG,
        "arquivo_config": B.CFG.get("_arquivo_config"),
        "configuracao_efetiva": {
            "seed": B.SEED, "dataset": B.CFG_DATASET["nome"], "num_tasks": num_tasks, "classes_por_tarefa": C,
            "classificador": tipo, "backbone": B.BACKBONE, "transformacao": B.CFG_DATASET.get("transformacao"),
            "embedding_dim": B.DIM_EMBEDDING, "perfil": B.CFG.get("_perfil"), "fecam": hp,
            "gauss_ca": None if tipo != "gauss_ca" else {"epocas": clf.epocas, "amostras_por_classe": clf.n_por_classe,
                                                         "lr": clf.lr, "momentum": clf.momentum, "lote": clf.lote,
                                                         "escala": clf.escala, "relu": clf.relu},
            "pseudo_negativos": B.CFG_PN,
        },
        "ambiente": {"torch": torch.__version__, "timm": B.VERSAO_TIMM,
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "particao_tarefas": {"classes_originais_por_tarefa": tarefas_orig,
                             "nomes_por_tarefa": [[nomes[mapa[c]] for c in t] for t in tarefas_orig]},
        "tarefas": {},
    }

    tempos = []
    for t in range(num_tasks):
        t0 = time.time()
        classes_novas = list(range(t * C, (t + 1) * C))
        classes_vistas = list(range((t + 1) * C))
        lt = loaders_treino[t]
        phi_t, y_t = lt.phi[lt.idx.to(lt.phi.device)], lt.y.to(B.DEVICE)
        medias = []
        for c in classes_novas:
            phi_c = phi_t[y_t == c]
            banco.adicionar_classe(c, phi_c)
            if tipo == "fecam":
                clf.adicionar_classe(c, phi_c)
            medias.append(phi_c.mean(0))
        info = {"classes": [nomes[c] for c in classes_novas]}
        if tipo == "gauss_ca":
            clf.adicionar_classes(torch.stack(medias))
            info["perda_realinhamento_por_epoca"] = clf.realinhar(banco, classes_vistas)
        tempos.append(time.time() - t0)
        info["tempo_treino_s"] = tempos[-1]
        json_data["tarefas"][str(t)] = info

        la = loaders_teste_acum[t]
        lg, lb = _avaliar(clf, la.phi[la.idx.to(la.phi.device)], la.y, classes_vistas, C)
        json_data.setdefault("acc_apos_tarefa", []).append(float((lg.argmax(1) == lb).mean() * 100))
        json_data.setdefault("acc_por_tarefa_apos_tarefa", []).append(B.acc_por_tarefa_vista(lg, lb, t + 1))
        if t >= 1:
            json_data.setdefault("metricas_vazamento_por_fronteira", {})[str(t)] = B.relatorio_vazamento(
                None, None, None, None, None, banco, None, f"após a tarefa {t}", logits_labels=(lg, lb))
        logging.info(f"[Tarefa {t}] {tipo}: acurácia no teste acumulado {json_data['acc_apos_tarefa'][-1]:.2f}% "
                     f"({tempos[-1]:.1f}s)")

    # resultado final
    acc_tarefas = json_data["acc_por_tarefa_apos_tarefa"][-1]
    json_data["resultado_final"] = {"acc_por_tarefa": acc_tarefas, "aac": float(np.mean(acc_tarefas)),
                                    "aac_incremental": float(np.mean(json_data["acc_apos_tarefa"]))}
    json_data["metricas_vazamento_final"] = B.relatorio_vazamento(
        None, None, None, None, None, banco, None, "final", logits_labels=(lg, lb))
    json_data["metricas_vazamento_final_padronizada"] = None   # uma única escala: não se aplica
    json_data["versao_principal"] = "original"
    json_data["esquecimento_medio"] = B.esquecimento_medio(json_data["acc_por_tarefa_apos_tarefa"])
    d = B.DIM_EMBEDDING
    b_banco = [4 * (banco.mu[c].numel() + banco.U[c].numel() + banco.e[c].numel() + 1)
               for c in range(B.NUM_CLASSES_TOTAIS)]
    json_data["memoria"] = ({"bytes_por_classe": clf.bytes_por_classe(d),
                             "descricao": "média + covariância completa por classe (FeCAM)"} if tipo == "fecam" else
                            {"bytes_por_classe": float(np.mean(b_banco)) + 4 * d,
                             "descricao": "banco de gaussianas + um vetor do classificador por classe"})
    json_data["tempos"] = {"cache_phi_s": tempo_cache, "treino_por_tarefa_s": tempos,
                           "total_s": time.time() - t_inicio}
    logging.info(f" > {tipo}: acurácia final {100 * json_data['metricas_vazamento_final']['acc']:.2f}% | "
                 f"AAC incremental {json_data['resultado_final']['aac_incremental']:.2f}% | "
                 f"esquecimento médio {json_data['esquecimento_medio']}")

    caminho = os.path.join(B.OS_RESULTS_DIR, f"results_{B.ID_EXECUCAO}.json")
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)
    logging.info(f" [JSON] {caminho}")
