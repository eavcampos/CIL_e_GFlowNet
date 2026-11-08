"""
Compara os braços LINHA DE BASE, GAUSS e EXPERIMENTO a partir dos JSON de cil_nova_estrategia_2.py.

Lê resultados/results/results_*.json, agrupa por (config, seed) e, se houver mais de uma execução da
mesma combinação, usa a mais recente. Configs "baseline*", "gauss*" e "experimento*" de mesmo sufixo
são comparados entre si (ex.: baseline_teste, gauss_teste e experimento_teste).

  baseline    -> sem pseudo-negativos
  gauss       -> pseudo-negativos sorteados às cegas (replay gaussiano já conhecido na literatura)
  experimento -> pseudo-negativos escolhidos pela margem de vazamento (modo "is")

O braço gauss é opcional: sem ele, a comparação é só linha de base x experimento. Diferenças
reportadas: gauss - linha de base (ganho do que já é conhecido) e experimento - gauss (ganho da
escolha pela margem); sem gauss, experimento - linha de base.

Gera em resultados/comparacao/<data_hora>/:
  resumo.csv                      uma linha por (seed, config) + médias e diferença pareada
  1_pares_por_seed.png            AAC, vazamento, erro dentro da tarefa e teto de ganho:
                                  cada seed é uma linha ligando linha de base -> experimento
  2_vazamento_por_fronteira.png   vazamento após cada tarefa (média e faixa entre seeds)
  3_acc_final_por_tarefa.png      acurácia final de cada tarefa (média e faixa entre seeds)
  4_diagnostico_pseudo_negativos.png  (só experimento) fração que vaza e ESS por época

Não precisa de PyTorch. Uso:
  python comparar_resultados.py                       # resultados/results -> resultados/comparacao
  python comparar_resultados.py --resultados PASTA --saida PASTA
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paleta categórica validada (dataviz, validate_palette.js: CVD e contraste aprovados)
# (#199e70 = verde-água um passo mais escuro, para contraste >= 3:1 no fundo branco)
COR = {"baseline": "#2a78d6", "experimento": "#eb6834", "gauss": "#199e70"}
ROTULO = {"baseline": "Linha de base (sem pseudo-negativos)",
          "gauss": "Gauss (pseudo-negativos às cegas)",
          "experimento": "Experimento (pseudo-negativos pela margem)"}
ROTULO_CURTO = {"baseline": "Linha\nde base", "gauss": "Gauss\n(às cegas)", "experimento": "Experimento\n(margem)"}
ORDEM_BRACOS = ["baseline", "gauss", "experimento"]   # do mais simples ao método proposto
TINTA, TINTA_2, GRADE = "#1f1f1e", "#5f5e58", "#e4e3df"
RAMPA_TAREFAS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]       # ordinal, uma matiz (azul)

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRADE, "axes.labelcolor": TINTA_2, "xtick.color": TINTA_2,
    "ytick.color": TINTA_2, "axes.titlecolor": TINTA, "axes.grid": True, "grid.color": GRADE,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": "white",
})


REFERENCIAS = ["ncm"]   # [Correções v2] classificador pela média de phi, sem treino (config_ncm_v2.json)


def papel(nome_config):
    for p in ORDEM_BRACOS + REFERENCIAS:
        if nome_config.startswith(p):
            return p, nome_config[len(p):]
    return None, None


def contrastes(bracos):
    """Pares (depois, antes) cuja diferença é reportada."""
    if "gauss" in bracos:
        return [("gauss", "baseline"), ("experimento", "gauss")]
    return [("experimento", "baseline")]


def carregar(pasta):
    """{sufixo: {papel: {seed: json}}}, mantendo a execução mais recente de cada combinação."""
    grupos, ids = {}, {}
    for caminho in sorted(glob.glob(os.path.join(pasta, "results_*.json"))):
        try:
            d = json.load(open(caminho, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print(f"  aviso: não consegui ler {caminho}")
            continue
        p, suf = papel(d.get("nome_config", ""))
        if p is None or "resultado_final" not in d:
            continue
        seed = d["configuracao_efetiva"]["seed"]
        chave = (suf, p, seed)
        if chave not in ids or d["id_execucao"] > ids[chave]:      # o id termina na data/hora
            ids[chave] = d["id_execucao"]
            grupos.setdefault(suf, {}).setdefault(p, {})[seed] = d
    return grupos


def metricas(d):
    mv = d.get("metricas_vazamento_final", {})
    pct = lambda v: None if v is None else 100 * v
    return {
        "aac": d["resultado_final"]["aac"],
        "vazamento": pct(mv.get("vazamento")),
        "erro_dentro_da_tarefa": pct(mv.get("erro_dentro_da_tarefa")),
        "teto_de_ganho": pct(mv.get("teto_de_ganho")),
        "acc_por_tarefa": d["resultado_final"]["acc_por_tarefa"],
        "aac_incremental": aac_incremental(d),
    }


def aac_incremental(d):
    """AAC da literatura: média das acurácias no teste acumulado após cada tarefa. Execuções novas
    gravam 'aac_incremental'; nas antigas, é reconstruída (tarefa 0 = primeira avaliação da tarefa 1,
    depois as acurácias de metricas_vazamento_por_fronteira)."""
    rf = d["resultado_final"]
    if rf.get("aac_incremental") is not None:
        return rf["aac_incremental"]
    try:
        a0 = d["tarefas"]["1"]["historico_treino"][0]["acc_por_tarefa"][0]
        fr = d["metricas_vazamento_por_fronteira"]
        return float(np.mean([a0] + [100 * fr[k]["acc"] for k in sorted(fr, key=int)]))
    except (KeyError, IndexError, TypeError):
        return None


NOMES_METRICA = {"aac": ("Acurácia média final", "%", "maior é melhor"),
                 "vazamento": ("Vazamento entre tarefas", "% das imagens de teste", "menor é melhor"),
                 "erro_dentro_da_tarefa": ("Erro dentro da tarefa", "% das imagens de teste", "menor é melhor"),
                 "teto_de_ganho": ("Teto de ganho (oráculo de tarefa)", "pontos percentuais", "menor é melhor")}


def salvar(fig, pasta, nome):
    caminho = os.path.join(pasta, nome)
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {caminho}")


def _texto_delta(dif, chave, nome):
    txt = f"{nome}: {dif.mean():+.2f}"
    if len(dif) > 1:
        txt += f" ± {dif.std(ddof=1):.2f}"
    melhor = (dif > 0) if chave == "aac" else (dif < 0)
    return txt + f" (melhor em {melhor.sum()}/{len(dif)})"


def grafico_pares(m, bracos, seeds, pasta, sufixo):
    k = len(bracos)
    fig, axes = plt.subplots(1, 4, figsize=(4.2 * 4, 4.6))
    for ax, chave in zip(axes, NOMES_METRICA):
        titulo, unidade, sentido = NOMES_METRICA[chave]
        v = {p: np.array([m[p][s][chave] for s in seeds], dtype=float) for p in bracos}
        for i, s in enumerate(seeds):
            ax.plot(range(k), [v[p][i] for p in bracos], color=GRADE, linewidth=2, zorder=1)
            ax.text(k - 1 + 0.08, v[bracos[-1]][i], str(s), fontsize=8, color=TINTA_2, va="center")
        for j, p in enumerate(bracos):
            ax.scatter(np.full(len(seeds), j), v[p], s=55, color=COR[p], zorder=3, edgecolor="white", linewidth=2)
        linhas = [_texto_delta(v[a] - v[b], chave, f"Δ {a[:5]}−{b[:5]}") for a, b in contrastes(bracos)]
        ax.set_title(titulo + "\n" + "\n".join(linhas), fontsize=9, loc="left")
        ax.set_xticks(range(k)); ax.set_xticklabels([ROTULO_CURTO[p] for p in bracos], fontsize=8)
        ax.set_xlim(-0.3, k - 1 + 0.4)
        ax.set_ylabel(f"{unidade} ({sentido})", fontsize=8)
        ax.grid(axis="x", visible=False)
    nomes = " × ".join({"baseline": "linha de base", "gauss": "gauss", "experimento": "experimento"}[p] for p in bracos)
    fig.suptitle(f"{nomes[0].upper() + nomes[1:]}, pareado por seed{sufixo}", x=0.01, ha="left",
                 fontsize=12, color=TINTA, fontweight="bold")
    fig.tight_layout()
    salvar(fig, pasta, "1_pares_por_seed.png")


def faixa(ax, x, valores, cor, rotulo, deslocamento=0.0):
    """Média com bigode do mínimo ao máximo entre seeds."""
    v = np.array(valores, dtype=float)                        # [seeds, pontos]
    media = np.nanmean(v, axis=0)
    xs = np.asarray(x, dtype=float) + deslocamento
    if v.shape[0] > 1:
        ax.vlines(xs, np.nanmin(v, axis=0), np.nanmax(v, axis=0), color=cor, linewidth=2, alpha=0.5)
    ax.plot(xs, media, "o-", color=cor, linewidth=2, markersize=8, markeredgecolor="white",
            markeredgewidth=2, label=rotulo)
    return media


def grafico_fronteiras(grupo, bracos, seeds, pasta, sufixo):
    fronteiras = sorted({int(t) for p in grupo.values() for d in p.values()
                         for t in d.get("metricas_vazamento_por_fronteira", {})})
    if not fronteiras:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, p in enumerate(bracos):
        desl = (i - (len(bracos) - 1) / 2) * 0.08
        vals = [[100 * grupo[p][s]["metricas_vazamento_por_fronteira"].get(str(t), {}).get("vazamento", np.nan)
                 for t in fronteiras] for s in seeds]
        media = faixa(ax, fronteiras, vals, COR[p], ROTULO[p], deslocamento=desl)
        ax.annotate(f"{media[-1]:.1f}%", (fronteiras[-1] + desl, media[-1]),
                    xytext=(10, 0), textcoords="offset points", va="center", fontsize=9, color=TINTA)
    ax.set_xticks(fronteiras); ax.set_xticklabels([f"após T{t}" for t in fronteiras])
    ax.set_ylabel("Vazamento (% das imagens de teste vistas até ali)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Vazamento após cada tarefa{sufixo}\n"
                 "ponto = média das seeds; barra vertical = mínimo a máximo", loc="left", fontsize=10)
    ax.legend(loc="upper left")
    fig.tight_layout()
    salvar(fig, pasta, "2_vazamento_por_fronteira.png")


def grafico_acc_tarefas(m, bracos, seeds, pasta, sufixo):
    n = len(m["baseline"][seeds[0]]["acc_por_tarefa"])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, p in enumerate(bracos):
        vals = [m[p][s]["acc_por_tarefa"] for s in seeds]
        faixa(ax, range(n), vals, COR[p], ROTULO[p], deslocamento=(i - (len(bracos) - 1) / 2) * 0.12)
    ax.set_xticks(range(n)); ax.set_xticklabels([f"T{t}" for t in range(n)])
    ax.set_xlabel("Tarefa (ordem de aprendizado)")
    ax.set_ylabel("Acurácia final na tarefa (%)")
    ax.set_title(f"Acurácia final de cada tarefa ao fim do treino{sufixo}\n"
                 "tarefas antigas à esquerda; ponto = média; barra = mínimo a máximo", loc="left", fontsize=10)
    ax.legend(loc="lower right")
    fig.tight_layout()
    salvar(fig, pasta, "3_acc_final_por_tarefa.png")


def grafico_diagnostico(grupo, seeds, pasta, sufixo):
    exp = grupo["experimento"]
    tarefas = sorted(int(t) for t in exp[seeds[0]]["tarefas"] if int(t) >= 1)
    if not tarefas or "pseudo_negativos" not in exp[seeds[0]]["tarefas"][str(tarefas[0])]["historico_treino"][0]:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    painéis = [("frac_vaza", "Pseudo-negativos que vazam (margem > 0)",
                "Deve CAIR: o Adapter novo aprende a recusar as classes antigas"),
               ("ess_frac", "Tamanho efetivo da amostragem / propostas",
                "Perto de 0 = regiões de vazamento na cauda: só então vale testar a GFlowNet")]
    for ax, (chave, titulo, leitura) in zip(axes, painéis):
        for j, t in enumerate(tarefas):
            curvas = []
            for s in seeds:
                h = exp[s]["tarefas"][str(t)]["historico_treino"]
                curvas.append([np.nan if r["pseudo_negativos"].get(chave) is None
                               else 100 * r["pseudo_negativos"][chave] for r in h])
            comp = min(len(c) for c in curvas)
            media = np.nanmean(np.array([c[:comp] for c in curvas]), axis=0)
            cor = RAMPA_TAREFAS[min(j, len(RAMPA_TAREFAS) - 1)]
            ax.plot(range(1, comp + 1), media, color=cor, linewidth=2, label=f"tarefa {t}")
            ax.annotate(f"T{t}", (comp, media[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=TINTA)
        ax.set_xlabel("Época de treino do Adapter da tarefa")
        ax.set_ylabel("% (média das seeds)")
        ax.set_ylim(bottom=0)
        ax.set_title(f"{titulo}\n{leitura}", loc="left", fontsize=10)
        ax.legend(loc="center left", bbox_to_anchor=(1.06, 0.5), fontsize=8)
    fig.suptitle(f"Diagnóstico dos pseudo-negativos durante o treino (só experimento){sufixo}",
                 x=0.01, ha="left", fontsize=12, color=TINTA, fontweight="bold")
    fig.tight_layout()
    salvar(fig, pasta, "4_diagnostico_pseudo_negativos.png")


def escrever_csv(m, bracos, seeds, pasta, referencias=None):
    referencias = referencias or {}
    caminho = os.path.join(pasta, "resumo.csv")
    chaves = list(NOMES_METRICA)
    n_tarefas = len(m["baseline"][seeds[0]]["acc_por_tarefa"])
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        fmt = lambda v, f: "" if v is None else format(v, f)
        w.writerow(["seed", "config"] + chaves + ["aac_incremental"] + [f"acc_T{t}" for t in range(n_tarefas)])
        for s in seeds:
            for p in bracos:
                w.writerow([s, p] + [f"{m[p][s][k]:.3f}" for k in chaves] + [fmt(m[p][s]["aac_incremental"], ".3f")]
                           + [f"{v:.3f}" for v in m[p][s]["acc_por_tarefa"]])
        for s, mr in sorted(referencias.items()):
            w.writerow([s, "ncm (referência, sem treino)"] + [fmt(mr[k], ".3f") for k in chaves]
                       + [fmt(mr["aac_incremental"], ".3f")] + [f"{v:.3f}" for v in mr["acc_por_tarefa"]])
        for p in bracos:
            w.writerow(["média", p] + [f"{np.mean([m[p][s][k] for s in seeds]):.3f}" for k in chaves])
        for a, b in contrastes(bracos):
            w.writerow([f"diferença média ({a} - {b})", ""] +
                       [f"{np.mean([m[a][s][k] - m[b][s][k] for s in seeds]):+.3f}" for k in chaves])
    print(f"  {caminho}")


def main():
    pasta_script = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Compara linha de base, gauss e experimento (Nova Estratégia 2)")
    ap.add_argument("--resultados", default=os.path.join(pasta_script, "resultados", "results"))
    ap.add_argument("--saida", default=os.path.join(pasta_script, "resultados", "comparacao"))
    args = ap.parse_args()

    grupos = carregar(args.resultados)
    if not grupos:
        print(f"Nenhum resultado de 'baseline*' ou 'experimento*' em {os.path.abspath(args.resultados)}.")
        return 1
    carimbo = f"{datetime.now():%Y%m%d_%H%M%S}"
    for sufixo, grupo in sorted(grupos.items()):
        nome_grupo = sufixo.strip("_") or "principal"
        if "baseline" not in grupo or "experimento" not in grupo:
            print(f"[{nome_grupo}] falta a linha de base ou o experimento; nada a comparar.")
            continue
        seeds_2 = sorted(set(grupo["baseline"]) & set(grupo["experimento"]))
        seeds_3 = sorted(set(seeds_2) & set(grupo.get("gauss", {})))
        comparacoes = []
        if seeds_3:
            comparacoes.append((["baseline", "gauss", "experimento"], seeds_3, ""))
        if seeds_2 and seeds_2 != seeds_3:
            # há seeds sem o braço gauss: compara também os dois braços originais em todas as seeds
            comparacoes.append((["baseline", "experimento"], seeds_2, "_2bracos" if seeds_3 else ""))
            if seeds_3:
                print(f"[{nome_grupo}] gauss só existe para {seeds_3}; os 3 braços são comparados nessas seeds, "
                      f"e linha de base x experimento também em {seeds_2}.")
        if not comparacoes:
            print(f"[{nome_grupo}] nenhuma seed com a linha de base e o experimento concluídos.")
            continue
        for bracos, seeds, extra in comparacoes:
            rotulo = f" ({nome_grupo})" if sufixo else ""
            pasta = os.path.join(args.saida, f"{carimbo}{sufixo}{extra}")
            os.makedirs(pasta, exist_ok=True)
            print(f"Comparando{rotulo} {bracos}: seeds {seeds} -> {os.path.abspath(pasta)}")
            m = {p: {s: metricas(grupo[p][s]) for s in seeds} for p in bracos}
            refs = {s: metricas(grupo["ncm"][s]) for s in seeds if s in grupo.get("ncm", {})}
            escrever_csv(m, bracos, seeds, pasta, refs)
            grafico_pares(m, bracos, seeds, pasta, rotulo)
            grafico_fronteiras(grupo, bracos, seeds, pasta, rotulo)
            grafico_acc_tarefas(m, bracos, seeds, pasta, rotulo)
            grafico_diagnostico(grupo, seeds, pasta, rotulo)
            for k, (titulo, _, _) in NOMES_METRICA.items():
                medias = " | ".join(f"{p[:5]} {np.mean([m[p][s][k] for s in seeds]):6.2f}" for p in bracos)
                difs = " | ".join(f"Δ {a[:5]}−{b[:5]} {np.mean([m[a][s][k] - m[b][s][k] for s in seeds]):+6.2f}"
                                  for a, b in contrastes(bracos))
                print(f"   {titulo:<34} {medias} | {difs}")
            ai = " | ".join(f"{p[:5]} {np.mean([m[p][s]['aac_incremental'] or np.nan for s in seeds]):6.2f}" for p in bracos)
            print(f"   {'AAC incremental':<34} {ai}")
            for s, mr in sorted(refs.items()):
                print(f"   Referência NCM (sem treino), seed {s}: acurácia final {mr['aac']:.2f}% | "
                      f"vazamento {mr['vazamento']:.2f}% | AAC incremental {mr['aac_incremental'] or float('nan'):.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
