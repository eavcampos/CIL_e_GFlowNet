#!/usr/bin/env python3
"""
[Fase B, adendo de 26/11/2026] Reavalia checkpoints JÁ SALVOS com a cabeça Mahalanobis (cabeca_mahalanobis.py),
sem treinar nada. Serve para medir o braço novo nas execuções de desenvolvimento (seeds 1993-1996) que já
existem, antes de decidir se ele entra no plano.

Cada checkpoint roda num processo separado, com o config gravado dentro dele (o script principal lê o config
uma vez, ao ser importado). A partição de classes, a divisão treino/validação e o cache de phi são os mesmos
da execução original; o script confere que as notas de cosseno reproduzem a acurácia original.

Uso (na pasta do projeto):
    python reavaliar_mahalanobis.py resultados/desenvolvimento/checkpoints/checkpoint_*_seed199[3-6]_*.pt
    python reavaliar_mahalanobis.py --saida resultados/reavaliacao_mahalanobis CHECKPOINT [CHECKPOINT ...]

Saída: resultados/reavaliacao_mahalanobis/reaval_mahalanobis_<execução>.json (um por checkpoint) e um
resumo no terminal.
"""
import argparse
import json
import os
import re
import subprocess
import sys

PASTA = os.path.dirname(os.path.abspath(__file__))
SEEDS_TESTE = set(range(1997, 2003))


def um_checkpoint(caminho, pasta_saida):
    import torch
    ck = torch.load(caminho, map_location="cpu", weights_only=False)
    cfg = dict(ck["configuracao"])
    nome_orig = os.path.splitext(os.path.basename(cfg.get("_arquivo_config", "config_desconhecido.json")))[0]
    if nome_orig.startswith("config_"):
        nome_orig = nome_orig[len("config_"):]
    pasta_cfg = os.path.join(pasta_saida, "configs")
    os.makedirs(pasta_cfg, exist_ok=True)
    cfg_tmp = os.path.join(pasta_cfg, f"config_{nome_orig}_reaval.json")
    with open(cfg_tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    del ck
    sys.argv = [os.path.join(PASTA, "cil_nova_estrategia_2.py"), "--config", cfg_tmp]
    sys.path.insert(0, PASTA)
    import cil_nova_estrategia_2 as base
    import cabeca_mahalanobis
    cabeca_mahalanobis.reavaliar(base, caminho, pasta_saida)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--saida", default=os.path.join(PASTA, "resultados", "reavaliacao_mahalanobis"))
    ap.add_argument("--um", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--permitir-seeds-de-teste", action="store_true",
                    help="Por padrão, checkpoints das seeds de teste (1997-2002) são recusados.")
    args = ap.parse_args()

    if args.um:
        um_checkpoint(args.checkpoints[0], args.saida)
        return

    feitos, falhas = [], []
    for ck in args.checkpoints:
        m = re.search(r"seed(\d{4})", os.path.basename(ck))
        if m and int(m.group(1)) in SEEDS_TESTE and not args.permitir_seeds_de_teste:
            print(f"[pulado] {ck}: seed de teste")
            continue
        if "adapter_compartilhado" in os.path.basename(ck):
            print(f"[pulado] {ck}: adapter_compartilhado (não se aplica)")
            continue
        print(f"\n=== {ck} ===", flush=True)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--um", "--saida", args.saida, ck])
        (feitos if r.returncode == 0 else falhas).append(ck)

    print("\n" + "=" * 100)
    print(f"{'execução':58s} {'cos':>6s} {'cos_p':>6s} {'mah':>6s} {'mah_p':>6s}")
    for ck in feitos:
        nome = os.path.splitext(os.path.basename(ck))[0].replace("checkpoint_", "")
        caminho = os.path.join(args.saida, f"reaval_mahalanobis_{nome}.json")
        try:
            with open(caminho, encoding="utf-8") as f:
                d = json.load(f)
            v, c = d["verificacao"], d["cabeca_mahalanobis"]
            print(f"{nome[:58]:58s} {v['acc_cosseno']:6.2f} {v['acc_cosseno_padronizada']:6.2f} "
                  f"{100 * c['metricas_vazamento_final']['acc']:6.2f} "
                  f"{100 * c['metricas_vazamento_final_padronizada']['acc']:6.2f}")
        except (OSError, KeyError, ValueError) as erro:
            print(f"{nome[:58]:58s} (sem resultado: {erro})")
    if falhas:
        print(f"\nFalharam ({len(falhas)}):")
        for ck in falhas:
            print("  " + ck)
    print("cos = cosseno (original); cos_p = cosseno padronizado; mah = cabeça Mahalanobis; mah_p = Mahalanobis "
          "padronizada (acurácia final, %)")


if __name__ == "__main__":
    main()
