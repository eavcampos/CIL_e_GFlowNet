"""
Pseudo-negativos no espaço de features do backbone congelado (phi), para a Estratégia 2.

POR QUE ISTO CABE NESTA ARQUITETURA
-----------------------------------
Na Estratégia 1, o Adapter de cada tarefa recebe phi = backbone(imagem), a saída da ResNet-18
CONGELADA (512 dimensões). Logo:
  * phi de uma classe antiga não muda nunca: as estatísticas guardadas ao fim de uma tarefa
    valem para sempre (é o que o drift semântico ~ 0 já mostra nos logs);
  * um phi sintético de uma classe antiga pode ser passado DIRETAMENTE pelos Adapters,
    antigos e novo, exatamente como um phi real. Não há nenhum "transporte" a aprender.

O QUE FALTA NO TREINO ATUAL
---------------------------
Na tarefa t, a perda CE usa só imagens das classes novas. Ela aprende "estas imagens são das
classes novas", mas nada ensina o Adapter_t a dar score BAIXO para imagens das classes antigas,
porque ele nunca as vê. É daí que vem o vazamento: uma imagem antiga recebe score alto de uma
classe nova.

O QUE ESTE MÓDULO FAZ
---------------------
1. BancoGaussianas: ao fim de cada tarefa, guarda para cada classe a média e a covariância
   (posto baixo + diagonal) de phi. Sem guardar imagens: continua sem replay de exemplos.
2. GeradorPseudoNegativos: durante a tarefa t, sorteia phi sintéticos das classes antigas,
   com o rótulo da classe antiga. Eles entram na mesma CE do treino (Estratégia 2), e o
   gradiente só pode mexer no Adapter_t e nas unidades LVQ da tarefa t: o único jeito de
   acertar é BAIXAR o score das classes novas para esses phi. É a direção que faltava.

   Três modos (config.json -> pseudo_negativos.modo):
   - "gauss": sorteio direto das gaussianas (replay gaussiano de features; mesma ideia de
              PASS, Zhu et al. CVPR 2021, e do alinhamento de classificador do SLCA,
              Zhang et al. ICCV 2023).
   - "is":    amostragem por importância: sorteia muitas propostas e reamostra com peso
              exp(beta * margem), em que margem = (melhor score de classe nova) - (score da
              própria classe antiga). Concentra os pseudo-negativos onde HOJE há vazamento.
              É o vencedor do brinquedo 2D (pseudo_negativos_gfn.py).
   - "gfn":   GFlowNet híbrida (V3 do brinquedo, pseudo_negativos_gfn_v2.py): escolhe a classe
              antiga (discreto), depois gera phi em poucos passos gaussianos (contínuo),
              treinada com trajectory balance e com um buffer alimentado pela amostragem por
              importância (trajetórias para trás). A recompensa muda à medida que o Adapter_t
              treina, então a GFlowNet é atualizada junto (Zhang et al., ICML 2022).
3. Diagnósticos por época: tamanho efetivo da amostra (ESS) da amostragem por importância e
   fração de pseudo-negativos que de fato vazam (margem > 0).

AVISO: escrito sem possibilidade de execução no ambiente em que foi produzido (PyTorch
indisponível). Rode primeiro uma configuração pequena (ver nota no config).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. Banco de gaussianas no espaço phi (backbone congelado)
# ============================================================================
class BancoGaussianas:
    """Para cada classe c: Sigma_c = U diag(e) U^T + s2 (I - U U^T).

    U: os `rank` autovetores principais da covariância; e: seus autovalores (+ encolhimento);
    s2: média dos autovalores restantes (+ encolhimento). Custo por classe: d*rank + rank + d,
    em vez de d*d (512*512 = 262 mil números por classe).
    """

    def __init__(self, dim, rank=32, encolhimento=1e-3, device="cpu"):
        self.dim, self.rank, self.enc, self.device = dim, rank, encolhimento, device
        self.mu, self.U, self.e, self.s2, self.n = {}, {}, {}, {}, {}

    @torch.no_grad()
    def adicionar_classe(self, c, phi):
        phi = phi.to(self.device, torch.float64)
        n, d = phi.shape
        mu = phi.mean(0)
        X = phi - mu
        cov = X.T @ X / max(n - 1, 1)
        evals, evecs = torch.linalg.eigh(cov)                       # ordem crescente
        evals, evecs = evals.flip(0).clamp(min=0), evecs.flip(1)
        r = max(1, min(self.rank, n - 1, d - 1))
        e = evals[:r] + self.enc
        s2 = (evals[r:].mean() if d > r else torch.tensor(0.0, dtype=evals.dtype)) + self.enc
        self.mu[c] = mu.float()
        self.U[c] = evecs[:, :r].float()
        self.e[c] = e.float()
        self.s2[c] = float(s2)
        self.n[c] = n

    def classes(self):
        return sorted(self.mu.keys())

    def media(self, c):
        return self.mu[c]

    def variancia_media(self, c):
        """Variância média por dimensão (traço / d)."""
        r = self.e[c].numel()
        return float((self.e[c].sum() + (self.dim - r) * self.s2[c]) / self.dim)

    @torch.no_grad()
    def amostrar(self, rotulos):
        """rotulos: LongTensor [m] de classes do banco -> phi sintéticos [m, d]."""
        out = torch.empty(len(rotulos), self.dim, device=self.device)
        for c in rotulos.unique().tolist():
            idx = (rotulos == c).nonzero(as_tuple=True)[0]
            m = len(idx)
            U, e, s2 = self.U[c], self.e[c], self.s2[c]
            eps_r = torch.randn(m, U.shape[1], device=self.device)
            eps_d = torch.randn(m, self.dim, device=self.device)
            eps_d = eps_d - (eps_d @ U) @ U.T                       # componente fora do subespaço
            out[idx] = self.mu[c] + (eps_r * e.sqrt()) @ U.T + math.sqrt(s2) * eps_d
        return out

    def log_prob(self, rotulos, x):
        """log N(x; mu_c, Sigma_c) para cada linha, com c = rotulos[i]. Diferenciável em x."""
        out = torch.empty(len(rotulos), device=x.device, dtype=x.dtype)
        for c in rotulos.unique().tolist():
            idx = (rotulos == c).nonzero(as_tuple=True)[0]
            U, e, s2 = self.U[c], self.e[c], self.s2[c]
            diff = x[idx] - self.mu[c]
            proj = diff @ U                                         # [m, r]
            resto = (diff ** 2).sum(1) - (proj ** 2).sum(1)
            maha = (proj ** 2 / e).sum(1) + resto / s2
            r = U.shape[1]
            logdet = torch.log(e).sum() + (self.dim - r) * math.log(s2)
            out[idx] = -0.5 * (maha + logdet + self.dim * math.log(2 * math.pi))
        return out

    def estado(self):
        return {"mu": {k: v.cpu() for k, v in self.mu.items()},
                "U": {k: v.cpu() for k, v in self.U.items()},
                "e": {k: v.cpu() for k, v in self.e.items()},
                "s2": dict(self.s2), "n": dict(self.n), "rank": self.rank, "encolhimento": self.enc}


@torch.no_grad()
def coletar_phi_por_classe(backbone, loader, classes, device):
    acc = {c: [] for c in classes}
    for images, labels in loader:
        phi = backbone(images.to(device))
        for c in classes:
            mask = labels.to(device) == c
            if mask.any():
                acc[c].append(phi[mask])
    return {c: torch.cat(v, 0) for c, v in acc.items() if v}


# ============================================================================
# 2. Scores: os mesmos da inferência (cosseno + LVQ), em versão de treino
# ============================================================================
def scores_combinados(phi, adapters_anteriores, adapter_novo, fc, lvq, classes_ativas,
                      classes_por_tarefa, alpha_lvq, tau_lvq, grad_novo):
    """Score combinado (1-alpha)*cos + alpha*LVQ para cada classe ativa, igual a
    calcular_logits_finais, mas:
      * usa lvq.prototipos_efetivos() (as unidades LVQ de tarefas antigas ficam congeladas);
      * os Adapters antigos rodam sem gradiente; o Adapter novo com gradiente se grad_novo.
    """
    with torch.no_grad():
        deltas = [ad(phi) for ad in adapters_anteriores]
    if grad_novo:
        deltas.append(adapter_novo(phi))
    else:
        with torch.no_grad():
            deltas.append(adapter_novo(phi))
    protos = lvq.prototipos_efetivos()
    cols = []
    for c in classes_ativas:
        delta = deltas[c // classes_por_tarefa]
        s_cos = F.cosine_similarity(delta, fc[c].unsqueeze(0).expand_as(delta), dim=1)
        idx = lvq.indices_por_classe[c]
        d2 = torch.cdist(delta, protos[idx]) ** 2
        s_lvq = torch.exp(-d2.min(dim=1).values / (2 * tau_lvq ** 2))
        cols.append((1 - alpha_lvq) * s_cos + alpha_lvq * s_lvq)
    return torch.stack(cols, dim=1)


def margem_vazamento(S, rotulos, classes_novas, classes_ativas):
    """margem = (maior score entre as classes novas) - (score da própria classe antiga).
    > 0 significa que este phi vazaria para uma classe nova."""
    pos = {c: i for i, c in enumerate(classes_ativas)}
    col_novas = torch.tensor([pos[c] for c in classes_novas], device=S.device)
    col_proprio = torch.tensor([pos[int(c)] for c in rotulos.tolist()], device=S.device)
    melhor_nova = S[:, col_novas].max(dim=1).values
    proprio = S.gather(1, col_proprio.unsqueeze(1)).squeeze(1)
    return melhor_nova - proprio


# ============================================================================
# 3. GFlowNet híbrida (classe antiga discreta -> phi contínuo), trajectory balance
# ============================================================================
class GFNHibrida(nn.Module):
    """s0 --(escolhe k, discreto)--> (k, t=0, z=mu_k) --(T passos gaussianos)--> (k, z_T) --> parar.

    Condições de Lahlou et al. (2023), seção G5 do documento:
      - absorção finita: T passos e parada obrigatória;
      - alcançabilidade: incrementos gaussianos com suporte em todo R^d;
      - densidade sobre estados com jacobiano 1: z_{t+1} = z_t + deriva + ruído;
      - P_B fixo: ponte browniana de volta a mu_k (escolha livre, seção G2.6).
    """

    def __init__(self, banco, classes_antigas, dim, passos=4, oculta=256, dim_emb=32):
        super().__init__()
        self.classes = list(classes_antigas)
        K = len(self.classes)
        self.T, self.dt, self.dim = passos, 1.0 / passos, dim
        self.register_buffer("mu", torch.stack([banco.media(c) for c in self.classes]))
        sig = torch.tensor([math.sqrt(banco.variancia_media(c)) for c in self.classes])
        self.register_buffer("sigma", sig)
        self.theta = nn.Parameter(torch.zeros(K))
        self.logZ = nn.Parameter(torch.zeros(()))
        self.emb = nn.Embedding(K, dim_emb)
        self.net = nn.Sequential(nn.Linear(dim + dim_emb + 1, oculta), nn.SiLU(),
                                 nn.Linear(oculta, oculta), nn.SiLU(),
                                 nn.Linear(oculta, 2 * dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)                          # início: passeio browniano puro

    def _passo(self, z, k, t):
        sig = self.sigma[k].unsqueeze(1)
        inp = torch.cat([(z - self.mu[k]) / sig, self.emb(k),
                         torch.full((len(k), 1), t / self.T, device=z.device)], dim=1)
        out = self.net(inp)
        deriva, log_s = out[:, :self.dim], out[:, self.dim:].clamp(-5.0, 3.0)
        media = z + deriva * sig * self.dt
        desvio = torch.exp(log_s) * sig * math.sqrt(self.dt)
        return media, desvio

    @torch.no_grad()
    def amostrar(self, n, ruido=1.0, uniforme=False):
        K = len(self.classes)
        p = torch.full((K,), 1.0 / K, device=self.mu.device) if uniforme else self.theta.softmax(0)
        k = torch.multinomial(p, n, replacement=True)
        z = self.mu[k].clone()
        traj = [z]
        for t in range(self.T):
            media, desvio = self._passo(z, k, t)
            z = media + ruido * desvio * torch.randn_like(z)
            traj.append(z)
        return k, traj

    @torch.no_grad()
    def para_tras(self, k, zT):
        """Trajetória sorteada com P_B (ponte browniana até mu_k), terminando em zT."""
        traj = [None] * (self.T + 1)
        traj[self.T] = zT
        mu, var = self.mu[k], (self.sigma[k] ** 2 * self.dt).unsqueeze(1)
        for t in range(self.T - 1, 0, -1):
            m = mu + (traj[t + 1] - mu) * t / (t + 1)
            traj[t] = m + torch.sqrt(var * t / (t + 1)) * torch.randn_like(zT)
        traj[0] = mu.clone()
        return traj

    def residuo_tb(self, k, traj, log_r):
        """delta = log Z + log P_F(k) + sum log p_F - log r - sum log p_B;  perda TB = delta^2."""
        log_pk = self.theta.log_softmax(0)[k]
        log_pf = torch.zeros(len(k), device=log_r.device)
        for t in range(self.T):
            media, desvio = self._passo(traj[t], k, t)
            log_pf = log_pf + (-(traj[t + 1] - media) ** 2 / (2 * desvio ** 2) - torch.log(desvio)
                               - 0.5 * math.log(2 * math.pi)).sum(1)
        mu, var = self.mu[k], (self.sigma[k] ** 2 * self.dt).unsqueeze(1)
        log_pb = torch.zeros_like(log_pf)
        for t in range(1, self.T):
            m = mu + (traj[t + 1] - mu) * t / (t + 1)
            v = var * t / (t + 1)
            log_pb = log_pb + (-(traj[t] - m) ** 2 / (2 * v)
                               - 0.5 * torch.log(2 * math.pi * v)).sum(1)
        return self.logZ + log_pk + log_pf - log_r - log_pb


# ============================================================================
# 4. Gerador: a interface que o script de treino usa
# ============================================================================
class GeradorPseudoNegativos:
    def __init__(self, banco, cfg, classes_por_tarefa, alpha_lvq, tau_lvq, device):
        self.banco, self.cfg = banco, cfg
        self.modo = str(cfg.get("modo", "is")).lower()
        self.beta = float(cfg.get("beta", 5.0))
        self.n_prop = int(cfg.get("n_proposta", 1024))
        self.relu = bool(cfg.get("relu", True))
        self.cpt, self.alpha, self.tau, self.device = classes_por_tarefa, alpha_lvq, tau_lvq, device
        self.cfg_gfn = cfg.get("gfn", {})
        self.gfn, self.opt_gfn = None, None
        self.reiniciar_estatisticas()

    # ---- estatísticas por época ------------------------------------------------
    def reiniciar_estatisticas(self):
        self.stats = {"ess_frac": [], "frac_vaza": [], "margem_media": [], "logZ_is": [],
                      "logZ_gfn": [], "perda_tb": []}

    def resumo_estatisticas(self):
        return {k: (float(sum(v) / len(v)) if v else None) for k, v in self.stats.items()}

    # ---- ciclo de vida por tarefa ----------------------------------------------
    def preparar_tarefa(self, classes_antigas):
        self.classes_antigas = list(classes_antigas)
        if self.modo == "gfn" and self.classes_antigas:
            g = self.cfg_gfn
            self.gfn = GFNHibrida(self.banco, self.classes_antigas, self.banco.dim,
                                  passos=int(g.get("passos", 4)), oculta=int(g.get("oculta", 256))
                                  ).to(self.device)
            self.opt_gfn = torch.optim.Adam([
                {"params": [p for n, p in self.gfn.named_parameters() if n != "logZ"],
                 "lr": float(g.get("lr", 1e-3))},
                {"params": [self.gfn.logZ], "lr": float(g.get("lr_logZ", 1e-1))}])
            self._logZ_iniciado = False

    def _pos(self, z):
        """phi da ResNet-18 sai de ReLU + avgpool: é não negativo. A gaussiana não é."""
        return F.relu(z) if self.relu else z

    @torch.no_grad()
    def _margem(self, z, rotulos, ctx):
        S = scores_combinados(self._pos(z), ctx["adapters_anteriores"], ctx["adapter_novo"], ctx["fc"],
                              ctx["lvq"], ctx["classes_ativas"], self.cpt, self.alpha, self.tau,
                              grad_novo=False)
        return margem_vazamento(S, rotulos, ctx["classes_novas"], ctx["classes_ativas"])

    @torch.no_grad()
    def _propostas_is(self, n_saida, ctx):
        """Propostas do banco (classe uniforme) reamostradas com peso exp(beta * margem)."""
        idx_cls = torch.randint(len(self.classes_antigas), (self.n_prop,), device=self.device)
        rot = torch.tensor(self.classes_antigas, device=self.device)[idx_cls]
        z = self.banco.amostrar(rot)
        m = self._margem(z, rot, ctx)
        logw = self.beta * m
        w = torch.softmax(logw, 0)
        ess = float(1.0 / (w ** 2).sum())
        self.stats["ess_frac"].append(ess / self.n_prop)
        self.stats["logZ_is"].append(float(torch.logsumexp(logw, 0) - math.log(self.n_prop)))
        sel = torch.multinomial(w, n_saida, replacement=True)
        return idx_cls[sel], rot[sel], z[sel]

    def _atualizar_gfn(self, ctx):
        g, gfn = self.cfg_gfn, self.gfn
        B = int(g.get("lote", 128))
        n_buf = int(B * float(g.get("fracao_buffer", 0.25)))
        n_on = (B - n_buf) // 2
        k1, t1 = gfn.amostrar(n_on, 1.0)
        k2, t2 = gfn.amostrar(B - n_buf - n_on, float(g.get("exploracao", 1.3)), uniforme=True)
        ks, trs = [k1, k2], [t1, t2]
        if n_buf > 0:
            k3, _, z3 = self._propostas_is(n_buf, ctx)
            ks.append(k3); trs.append(gfn.para_tras(k3, z3))
        k = torch.cat(ks)
        traj = [torch.cat(p, 0) for p in zip(*trs)]
        with torch.no_grad():
            rot = torch.tensor(self.classes_antigas, device=self.device)[k]
            log_r = (-math.log(len(self.classes_antigas)) + self.banco.log_prob(rot, traj[-1])
                     + self.beta * self._margem(traj[-1], rot, ctx))
            if not self._logZ_iniciado:
                # log Z = log E_{p_antiga}[exp(beta*margem)]: a estimativa da amostragem por
                # importância é o ponto de partida natural (o banco já é normalizado).
                gfn.logZ.fill_(self.stats["logZ_is"][-1] if self.stats["logZ_is"] else 0.0)
                self._logZ_iniciado = True
        delta = gfn.residuo_tb(k, traj, log_r)
        perda = (delta ** 2).mean()
        self.opt_gfn.zero_grad()
        perda.backward()
        torch.nn.utils.clip_grad_norm_(gfn.parameters(), 10.0)
        self.opt_gfn.step()
        self.stats["perda_tb"].append(float(perda))
        self.stats["logZ_gfn"].append(float(gfn.logZ))

    # ---- chamada principal -----------------------------------------------------
    def amostrar(self, n, ctx):
        """Devolve (phi_sintetico [n, d], rotulo_antigo [n]) para a CE da tarefa atual."""
        if not self.classes_antigas:
            return None, None
        dev = self.device
        if self.modo == "gauss":
            idx = torch.randint(len(self.classes_antigas), (n,), device=dev)
            rot = torch.tensor(self.classes_antigas, device=dev)[idx]
            z = self.banco.amostrar(rot)
        elif self.modo == "is":
            _, rot, z = self._propostas_is(n, ctx)
        elif self.modo == "gfn":
            for _ in range(int(self.cfg_gfn.get("atualizacoes_por_lote", 1))):
                self._atualizar_gfn(ctx)
            k, traj = self.gfn.amostrar(n, 1.0)
            rot = torch.tensor(self.classes_antigas, device=dev)[k]
            z = traj[-1]
        else:
            raise ValueError(f"pseudo_negativos.modo desconhecido: {self.modo!r}")
        m = self._margem(z, rot, ctx)
        self.stats["frac_vaza"].append(float((m > 0).float().mean()))
        self.stats["margem_media"].append(float(m.mean()))
        return self._pos(z).detach(), rot.detach()
