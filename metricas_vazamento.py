"""
Métricas de vazamento entre tasks em class-incremental learning (CIL).

Entrada mínima (numa etapa t, avaliando todas as classes vistas até ela):
    logits      [n, C]  score de cada amostra para cada classe vista (ex.: cosseno com protótipos)
    y_true      [n]     classe verdadeira (0..C-1)
    class_task  [C]     task de cada classe (0..T-1)

Entrada opcional:
    protos      [C, d]  protótipos no espaço CONGELADO (para a similaridade entre tasks)
    task_stats  [(mu_k, sd_k)]  média e desvio do score vencedor das amostras de treino da task k,
                         guardados ao fim da task k (compatível com o protocolo sem replay)

Definições (ver documento):
    t_hat(x) = task da classe prevista,  tau(x) = task verdadeira
    vazamento(x) = 1[t_hat(x) != tau(x)]
    1 - Acc = Lambda + E_WP   (decomposição exata: classes de tasks distintas são disjuntas)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# utilidades
# ---------------------------------------------------------------------------
def _rank(a: np.ndarray) -> np.ndarray:
    """Postos com empates resolvidos pela média (para Spearman)."""
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    return (sums / counts)[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def partial_spearman(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Correlação de Spearman entre a e b controlando c (resíduos de regressão linear nos postos)."""
    ra, rb, rc = _rank(a), _rank(b), _rank(c)
    X = np.column_stack([np.ones_like(rc), rc])
    res_a = ra - X @ np.linalg.lstsq(X, ra, rcond=None)[0]
    res_b = rb - X @ np.linalg.lstsq(X, rb, rcond=None)[0]
    if res_a.std() == 0 or res_b.std() == 0:
        return float("nan")
    return float(np.corrcoef(res_a, res_b)[0, 1])


# ---------------------------------------------------------------------------
# resultado
# ---------------------------------------------------------------------------
@dataclass
class LeakageReport:
    acc: float                   # acurácia CIL
    leakage: float               # Lambda: fração de amostras roteadas para a task errada
    leakage_bal: float           # Lambda balanceado: 1 - média da diagonal de L
    e_wp: float                  # erro dentro da task, com a task certa
    leak_share: float            # fração do erro total devida ao vazamento
    acc_oracle: float            # acurácia com oráculo de task
    gain_ceiling: float          # teto de ganho = acc_oracle - acc
    L: np.ndarray                # matriz de vazamento normalizada por linha [T, T]
    direction: float             # índice de direção D (massa para tasks novas / para antigas)
    offset_profile: dict         # rho(k): vazamento médio por deslocamento k = j - i
    rho_recency: float           # Spearman(L_ij, j - i)
    rho_semantic: float = float("nan")          # Spearman(L_ij, S_ij)
    rho_semantic_partial: float = float("nan")  # idem, controlando j - i
    S: np.ndarray | None = None                 # similaridade entre tasks [T, T]
    leakage_norm: float = float("nan")          # Lambda após normalizar scores por task
    calib_share: float = float("nan")           # (Lambda - Lambda_norm) / Lambda
    counts: np.ndarray = field(default=None)    # matriz de contagens [T, T]

    def resumo(self) -> str:
        T = self.L.shape[0]
        linhas = [
            f"Acurácia CIL ................ {self.acc:7.2%}",
            f"Vazamento (Lambda) .......... {self.leakage:7.2%}",
            f"Vazamento balanceado ........ {self.leakage_bal:7.2%}",
            f"Erro dentro da task (E_WP) .. {self.e_wp:7.2%}",
            f"Parcela do erro por vazamento {self.leak_share:7.2%}",
            f"Acurácia com oráculo de task  {self.acc_oracle:7.2%}",
            f"Teto de ganho ............... {self.gain_ceiling * 100:6.2f} p.p.",
            f"Índice de direção D ......... {self.direction:7.3f}",
            f"Spearman(L, j-i) recência ... {self.rho_recency:7.3f}",
        ]
        if not np.isnan(self.rho_semantic):
            linhas += [
                f"Spearman(L, S) semântica .... {self.rho_semantic:7.3f}",
                f"  parcial, controlando j-i .. {self.rho_semantic_partial:7.3f}",
            ]
        if not np.isnan(self.leakage_norm):
            linhas += [
                f"Vazamento após normalização . {self.leakage_norm:7.2%}",
                f"Parcela atribuível a escala . {self.calib_share:7.2%}",
            ]
        linhas.append("Perfil por deslocamento k = j - i:")
        for k in sorted(self.offset_profile):
            linhas.append(f"  k = {k:+d}: {self.offset_profile[k]:.3f}")
        linhas.append("Matriz L (linha = task verdadeira, coluna = task prevista):")
        for i in range(T):
            linhas.append("  " + " ".join(f"{v:5.3f}" for v in self.L[i]))
        return "\n".join(linhas)


# ---------------------------------------------------------------------------
# métricas
# ---------------------------------------------------------------------------
def task_similarity(protos: np.ndarray, class_task: np.ndarray, T: int) -> np.ndarray:
    """S_ij = média, sobre as classes a da task i, da maior similaridade de cosseno
    entre o protótipo de a e os protótipos das classes da task j. Assimétrica."""
    P = protos / np.linalg.norm(protos, axis=1, keepdims=True)
    cos = P @ P.T
    S = np.zeros((T, T))
    for i in range(T):
        ci = np.where(class_task == i)[0]
        for j in range(T):
            if i == j:
                continue
            cj = np.where(class_task == j)[0]
            S[i, j] = cos[np.ix_(ci, cj)].max(axis=1).mean()
    return S


def normalize_by_task(logits: np.ndarray, class_task: np.ndarray, task_stats) -> np.ndarray:
    """Padroniza os scores de cada task com (mu_k, sd_k) obtidos no TREINO da task k."""
    out = logits.astype(float).copy()
    for k, (mu, sd) in enumerate(task_stats):
        cols = class_task == k
        out[:, cols] = (out[:, cols] - mu) / max(sd, 1e-8)
    return out


def leakage_metrics(logits, y_true, class_task, protos=None, task_stats=None) -> LeakageReport:
    logits = np.asarray(logits, dtype=float)
    y_true = np.asarray(y_true)
    class_task = np.asarray(class_task)
    T = int(class_task.max()) + 1
    n = len(y_true)

    y_pred = logits.argmax(1)
    tau = class_task[y_true]
    t_hat = class_task[y_pred]

    correct = y_pred == y_true
    routed_ok = t_hat == tau
    acc = correct.mean()
    leakage = (~routed_ok).mean()
    e_wp = (routed_ok & ~correct).mean()
    err = 1 - acc
    assert abs(err - (leakage + e_wp)) < 1e-9, "decomposição 1 - Acc = Lambda + E_WP falhou"

    # oráculo de task: argmax restrito às classes da task verdadeira
    masked = np.where(class_task[None, :] == tau[:, None], logits, -np.inf)
    acc_oracle = (masked.argmax(1) == y_true).mean()

    # matriz de vazamento
    counts = np.zeros((T, T), dtype=int)
    np.add.at(counts, (tau, t_hat), 1)
    row = counts.sum(1, keepdims=True)
    L = np.divide(counts, row, out=np.zeros_like(counts, dtype=float), where=row > 0)

    iu = np.triu_indices(T, 1)
    il = np.tril_indices(T, -1)
    fwd, bwd = L[iu].sum(), L[il].sum()
    direction = fwd / bwd if bwd > 0 else float("inf")

    offset_profile = {}
    for k in range(-(T - 1), T):
        if k == 0:
            continue
        vals = [L[i, i + k] for i in range(T) if 0 <= i + k < T]
        offset_profile[k] = float(np.mean(vals))

    off = ~np.eye(T, dtype=bool)
    Lv = L[off]
    dist = (np.arange(T)[None, :] - np.arange(T)[:, None])[off].astype(float)
    rep = LeakageReport(
        acc=acc, leakage=leakage, leakage_bal=1 - np.diag(L).mean(), e_wp=e_wp,
        leak_share=leakage / err if err > 0 else 0.0,
        acc_oracle=acc_oracle, gain_ceiling=acc_oracle - acc,
        L=L, direction=direction, offset_profile=offset_profile,
        rho_recency=spearman(Lv, dist), counts=counts,
    )

    if protos is not None:
        S = task_similarity(np.asarray(protos, dtype=float), class_task, T)
        rep.S = S
        rep.rho_semantic = spearman(Lv, S[off])
        rep.rho_semantic_partial = partial_spearman(Lv, S[off], dist)

    if task_stats is not None:
        z = normalize_by_task(logits, class_task, task_stats)
        t_hat_n = class_task[z.argmax(1)]
        rep.leakage_norm = (t_hat_n != tau).mean()
        rep.calib_share = (leakage - rep.leakage_norm) / leakage if leakage > 0 else 0.0

    return rep


def leakage_over_steps(per_step: list[dict]) -> list[LeakageReport]:
    """Aplica leakage_metrics a cada etapa t. Cada dict tem as chaves de leakage_metrics."""
    return [leakage_metrics(**d) for d in per_step]


# ---------------------------------------------------------------------------
# demonstração com dados sintéticos
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T, C_per, d, n_per = 5, 10, 64, 40
    C = T * C_per
    class_task = np.repeat(np.arange(T), C_per)

    # protótipos no espaço congelado; tasks 0 e 3 recebem classes parecidas (confusão semântica)
    protos = rng.normal(size=(C, d))
    protos[class_task == 3] = protos[class_task == 0] + 0.6 * rng.normal(size=(C_per, d))

    y_true = np.repeat(np.arange(C), n_per)
    feats = protos[y_true] + 1.2 * rng.normal(size=(len(y_true), d))
    P = protos / np.linalg.norm(protos, axis=1, keepdims=True)
    F = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    logits = F @ P.T

    # adapters mais novos produzem scores inflados (viés de recência por escala)
    logits = logits + 0.03 * class_task[None, :]

    # estatísticas "de treino" por task: aqui aproximadas no próprio conjunto, só para a demo
    tau = class_task[y_true]
    stats = []
    for k in range(T):
        s = logits[tau == k][:, class_task == k].max(1)
        stats.append((s.mean(), s.std()))

    rep = leakage_metrics(logits, y_true, class_task, protos=protos, task_stats=stats)
    print(rep.resumo())
