"""
[Fase B, adendo de 26/11/2026] Cabeça Mahalanobis sobre as saídas dos Adapters por tarefa.

Ideia: o FeCAM (braço fecam) mede, para cada classe, a forma e a escala da nuvem de pontos (covariância) e
compara as distâncias levando isso em conta. Uma distância de Mahalanobis é medida "em desvios da própria
classe", então tende a ser comparável entre classes, mesmo que cada tarefa use um espaço diferente. Aqui a
mesma receita do FeCAM é aplicada às saídas do Adapter de cada tarefa, em vez de ser aplicada a phi:

  ajuste (fim da tarefa k, só com as imagens de TREINO da tarefa k, sem a parte de validação):
      z = normaliza(Adapter_k(phi));  para cada classe c da tarefa k: média e covariância de z,
      encolhimento (alpha1, alpha2) e normalização para correlação (iguais ao braço fecam), pseudo-inversa;
  nota da classe c (tarefa k) para uma imagem: -distância de Mahalanobis entre normaliza(Adapter_k(phi)) e c.

Também é reportada a versão padronizada: (nota - mu_k) / sigma_k, com mu_k e sigma_k da nota vencedora nas
imagens de treino da tarefa k (mesma regra da padronização dos cossenos).

Não muda o treino: é uma avaliação a mais, calculada em todo braço com Adapters por tarefa (inclusive o NCM,
em que o Adapter é a identidade e a cabeça coincide com o FeCAM sem a transformação de Tukey). Não se aplica
ao adapter_compartilhado (o Adapter muda depois do fim de cada tarefa). Hiperparâmetros: os do braço fecam
para o mesmo backbone e dataset, sem Tukey (as saídas dos Adapters podem ser negativas); nada é ajustado.

Dois usos:
  1) dentro de cil_nova_estrategia_2.py (chave treino.cabeca_mahalanobis, padrão true): grava o bloco
     "cabeca_mahalanobis" no JSON de cada execução;
  2) reavaliar(base, checkpoint, pasta): recalcula tudo a partir de um checkpoint já salvo, sem treinar
     (chamado pelo reavaliar_mahalanobis.py).
"""
import os
import json
import time
import logging

import numpy as np
import torch
import torch.nn.functional as F

from bracos_sem_adapter import hiperparametros_fecam, _encolher_e_normalizar


def hiperparametros(base):
    hp = dict(hiperparametros_fecam(base))
    hp["tukey_beta"] = None
    hp.update(base.CFG.get("cabeca_mahalanobis_hp", {}) or {})
    return hp


@torch.no_grad()
def _phi_e_rotulos(backbone, loader, device):
    """phi e rótulos CIL de todas as imagens de um loader (LoaderPhi ou DataLoader comum)."""
    if hasattr(loader, "phi") and hasattr(loader, "idx"):
        return loader.phi[loader.idx.to(loader.phi.device)].to(device).float(), loader.y.clone().cpu()
    fs, ys = [], []
    for x, y in loader:
        fs.append(backbone(x.to(device)).float())
        ys.append(torch.as_tensor(y).cpu())
    return torch.cat(fs), torch.cat(ys)


class CabecaMahalanobis:
    def __init__(self, base):
        self.B = base
        self.C = base.CLASSES_POR_TAREFA
        self.dev = base.DEVICE
        self.hp = hiperparametros(base)
        self.medias = []      # por tarefa: [C, d] float32 no dispositivo
        self.inversas = []    # por tarefa: [C, d, d] float32 na CPU
        self.stats = []       # por tarefa: (mu_k, sigma_k) da nota vencedora no treino da tarefa
        self.registro = {"acc_apos_tarefa": [], "acc_apos_tarefa_padronizada": [],
                         "acc_por_tarefa_apos_tarefa": [], "acc_por_tarefa_apos_tarefa_padronizada": []}
        self.tempo_s = 0.0

    @torch.no_grad()
    def _z(self, adapter, phi):
        return F.normalize(adapter(phi.float()).double(), dim=1, eps=1e-8)

    @torch.no_grad()
    def _notas_tarefa(self, k, z):
        """-Mahalanobis de cada imagem (z já no espaço do Adapter k) para as C classes da tarefa k."""
        z = z.float()
        mus, invs = self.medias[k], self.inversas[k]
        saida = torch.empty(z.shape[0], self.C, device=self.dev)
        for j in range(self.C):
            A = invs[j].to(self.dev, non_blocking=True)
            for i in range(0, z.shape[0], 8192):
                dif = z[i:i + 8192] - mus[j]
                saida[i:i + 8192, j] = -((dif @ A) * dif).sum(1)
        return saida

    @torch.no_grad()
    def ajustar_tarefa(self, k, adapter, backbone, loader_treino):
        t0 = time.time()
        phi, y = _phi_e_rotulos(backbone, loader_treino, self.dev)
        y = y.to(self.dev)
        z = self._z(adapter, phi)
        mus, invs = [], []
        for c in range(k * self.C, (k + 1) * self.C):
            x = z[y == c]
            if x.shape[0] == 0:
                raise ValueError(f"[Cabeça Mahalanobis] classe {c} sem imagens de treino na tarefa {k}.")
            cov = torch.cov(x.T) if x.shape[0] > 1 else torch.zeros(x.shape[1], x.shape[1], dtype=x.dtype,
                                                                     device=x.device)
            cov = _encolher_e_normalizar(cov, self.hp["alpha1"], self.hp["alpha2"])
            mus.append(x.mean(0).float())
            invs.append(torch.linalg.pinv(cov).float().cpu())
        self.medias.append(torch.stack(mus))
        self.inversas.append(torch.stack(invs))
        vencedora = self._notas_tarefa(k, z).max(1).values
        self.stats.append((float(vencedora.mean()), float(vencedora.std()) + 1e-8))
        self.tempo_s += time.time() - t0

    @torch.no_grad()
    def notas(self, adapters, phi):
        return torch.cat([self._notas_tarefa(k, self._z(adapters[k], phi)) for k in range(len(adapters))], dim=1)

    @torch.no_grad()
    def _coletar(self, adapters, backbone, loader):
        phi, y = _phi_e_rotulos(backbone, loader, self.dev)
        return self.notas(adapters, phi).cpu().numpy().astype(float), y.numpy()

    def registrar_etapa(self, adapters, backbone, loader_acumulado, t):
        """Avaliação no teste acumulado logo após a tarefa t (adapters = Adapters das tarefas 0..t)."""
        t0 = time.time()
        S, lb = self._coletar(adapters[:t + 1], backbone, loader_acumulado)
        Z = self.B.normalize_by_task(S, np.arange(S.shape[1]) // self.C, self.stats)
        r = self.registro
        r["acc_apos_tarefa"].append(float((S.argmax(1) == lb).mean() * 100))
        r["acc_apos_tarefa_padronizada"].append(float((Z.argmax(1) == lb).mean() * 100))
        r["acc_por_tarefa_apos_tarefa"].append(self.B.acc_por_tarefa_vista(S, lb, t + 1))
        r["acc_por_tarefa_apos_tarefa_padronizada"].append(self.B.acc_por_tarefa_vista(Z, lb, t + 1))
        self.tempo_s += time.time() - t0
        logging.info(f"[Tarefa {t}] Cabeça Mahalanobis: acurácia no teste acumulado {r['acc_apos_tarefa'][-1]:.2f}% "
                     f"(padronizada: {r['acc_apos_tarefa_padronizada'][-1]:.2f}%)")

    def finalizar(self, adapters, backbone, loader_total, banco):
        S, lb = self._coletar(adapters, backbone, loader_total)
        B = self.B
        r = dict(self.registro)
        res = {
            "hiperparametros": self.hp,
            "dados_de_ajuste": "imagens de treino da própria tarefa (sem a parte de validação), no fim da tarefa",
            **r,
            "metricas_vazamento_final": B.relatorio_vazamento(
                None, None, None, None, None, banco, self.stats, "final — cabeça Mahalanobis", logits_labels=(S, lb)),
            "metricas_vazamento_final_padronizada": B.relatorio_vazamento(
                None, None, None, None, None, banco, self.stats, "final — cabeça Mahalanobis", padronizar=True,
                logits_labels=(S, lb)),
            "estatisticas_score_por_tarefa": [[float(m), float(s)] for m, s in self.stats],
            "dispersao_escala_DE": B.dispersao_escala(self.stats),
            "esquecimento_medio": B.esquecimento_medio(r["acc_por_tarefa_apos_tarefa"]),
            "esquecimento_medio_padronizado": B.esquecimento_medio(r["acc_por_tarefa_apos_tarefa_padronizada"]),
            "bytes_por_classe": 4 * (B.DIM_EMBEDDING + B.DIM_EMBEDDING ** 2),
            "tempo_s": self.tempo_s,
        }
        logging.info(f" > [Cabeça Mahalanobis] acurácia final {100 * res['metricas_vazamento_final']['acc']:.2f}% | "
                     f"padronizada {100 * res['metricas_vazamento_final_padronizada']['acc']:.2f}% | "
                     f"DE {res['dispersao_escala_DE']} | {self.tempo_s:.0f}s")
        return res


# ---------------------------------------------------------------------------------------------------------
# Reavaliação de um checkpoint já salvo (sem treinar)
# ---------------------------------------------------------------------------------------------------------
def _restaurar_banco(B, estado):
    banco = B.BancoGaussianas(B.DIM_EMBEDDING, rank=int(estado.get("rank", 32)),
                              encolhimento=float(estado.get("encolhimento", 1e-3)), device=B.DEVICE)
    banco.mu = {int(k): v.to(B.DEVICE) for k, v in estado["mu"].items()}
    banco.U = {int(k): v.to(B.DEVICE) for k, v in estado["U"].items()}
    banco.e = {int(k): v.to(B.DEVICE) for k, v in estado["e"].items()}
    banco.s2 = {int(k): float(v) for k, v in estado["s2"].items()}
    banco.n = {int(k): int(v) for k, v in estado["n"].items()}
    return banco


def reavaliar(base, caminho_checkpoint, pasta_saida):
    B = base
    t_inicio = time.time()
    ck = torch.load(caminho_checkpoint, map_location="cpu", weights_only=False)
    if bool(B.CFG_TREINO.get("adapter_compartilhado", False)):
        raise SystemExit("A cabeça Mahalanobis não se aplica ao adapter_compartilhado.")
    C = B.CLASSES_POR_TAREFA
    n_tarefas = len(ck["adapters_state_dicts"])
    B.NUM_TASKS, B.NUM_CLASSES_TOTAIS = n_tarefas, n_tarefas * C
    logging.info(f"=== [Reavaliação, cabeça Mahalanobis] {os.path.basename(caminho_checkpoint)} — {n_tarefas} tarefas, "
                 f"seed {B.SEED} ===")

    train_set, test_set = B.carregar_dataset_base(B.CFG_DATASET)
    tarefas = ck["tarefas_classes_originais"]
    recalculada = B.montar_particao_tarefas(len(train_set.classes), n_tarefas, C, B.SEED, ordem=B.ORDEM_CLASSES)
    if [list(t) for t in recalculada] != [list(t) for t in tarefas]:
        raise SystemExit("A partição de classes recalculada não bate com a do checkpoint; reavaliação abortada.")
    mapa, nomes = B.preparar_particao_e_nomes(train_set, tarefas)

    backbone_real = B.criar_backbone_pretreinado_congelado()
    if B.CACHE_PHI:
        phi_tr, phi_te = B.preparar_cache_phi(backbone_real, train_set, test_set, B.CFG_DATASET)
        train_set = B.DatasetPhi(phi_tr.to(B.DEVICE), train_set.targets, train_set.classes)
        test_set = B.DatasetPhi(phi_te.to(B.DEVICE), test_set.targets, test_set.classes)
        backbone = B.BackboneIdentidade().to(B.DEVICE)
        del backbone_real
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        backbone = backbone_real
    loaders_treino, _, loaders_teste_acum, _, _ = B.obter_loaders_cil(
        train_set, test_set, tarefas, mapa, B.BATCH_SIZE,
        fracao_validacao=max(B.PA_FRACAO_VAL, B.FRACAO_EST_VAL), seed=B.SEED)

    adapters = []
    for sd in ck["adapters_state_dicts"]:
        ad = B.Adapter(dim=int(ck["dim_embedding"]), r=int(ck["r_adapter"]))
        ad.load_state_dict(sd)
        ad = ad.to(B.DEVICE)
        B.congelar_adapter(ad)
        adapters.append(ad)
    fc = ck["fc"].to(B.DEVICE)
    lvq = None
    if B.ALPHA_LVQ != 0:
        linhas, colunas = B.dimensionar_grade_lvq(n_tarefas, C, B.UNIDADES_LVQ_POR_CLASSE)
        lvq = B.criar_mapa_lvq(linhas, colunas, B.DIM_EMBEDDING, B.NUM_CLASSES_TOTAIS, n_tarefas, C)
        lvq.load_state_dict(ck["lvq_state_dict"])
    banco = _restaurar_banco(B, ck["banco_gaussianas"])
    task_stats = [tuple(s) for s in ck["estatisticas_score_por_tarefa"]]

    # verificação: as notas de cosseno do checkpoint devem reproduzir a acurácia da execução original
    lg, lb = B.coletar_logits(backbone, adapters, fc, lvq, loaders_teste_acum[-1])
    zg = B.normalize_by_task(lg, np.arange(lg.shape[1]) // C, task_stats)
    verif = {"acc_cosseno": float((lg.argmax(1) == lb).mean() * 100),
             "acc_cosseno_padronizada": float((zg.argmax(1) == lb).mean() * 100)}
    logging.info(f"[Verificação] cosseno, recalculado do checkpoint: {verif['acc_cosseno']:.2f}% "
                 f"(padronizado: {verif['acc_cosseno_padronizada']:.2f}%)")

    cabeca = CabecaMahalanobis(B)
    for t in range(n_tarefas):
        cabeca.ajustar_tarefa(t, adapters[t], backbone, loaders_treino[t])
        cabeca.registrar_etapa(adapters, backbone, loaders_teste_acum[t], t)
    res = cabeca.finalizar(adapters, backbone, loaders_teste_acum[-1], banco)

    nome = os.path.splitext(os.path.basename(caminho_checkpoint))[0].replace("checkpoint_", "")
    saida = {
        "checkpoint": os.path.abspath(caminho_checkpoint), "execucao_original": nome, "seed": B.SEED,
        "dataset": B.CFG_DATASET["nome"], "backbone": B.BACKBONE, "num_tasks": n_tarefas, "classes_por_tarefa": C,
        "pseudo_negativos": B.CFG_PN.get("modo"), "padronizar_notas": B.PADRONIZAR_NOTAS,
        "verificacao": verif, "cabeca_mahalanobis": res, "tempo_total_s": time.time() - t_inicio,
    }
    os.makedirs(pasta_saida, exist_ok=True)
    caminho = os.path.join(pasta_saida, f"reaval_mahalanobis_{nome}.json")
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(saida, f, indent=4, ensure_ascii=False)
    logging.info(f" [JSON] {caminho}")
    return caminho
