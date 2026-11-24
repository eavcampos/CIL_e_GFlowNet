#!/usr/bin/env python3
"""
Separa os resultados de DESENVOLVIMENTO dos de TESTE (Fase A5 do plano; Seção 1 do PLANO_DE_ANALISE.md).

- Seeds de desenvolvimento: 1993-1996 -> resultados/desenvolvimento/<subpasta>/
- Seeds de teste (1997 em diante)     -> resultados/teste/<subpasta>/

Subpastas tratadas: results, logs, checkpoints, figures_cil (as que existirem). execucoes/ não é
mexida, porque o rodar_experimentos.sh usa os marcadores de lá para retomar baterias.
Os arquivos são reconhecidos pelo trecho "seedNNNN" no nome.

resultados/comparacao/ (comparações antigas, que misturam seeds de desenvolvimento e teste)
vai inteira para resultados/desenvolvimento/comparacao/, porque não serve como análise de teste.

Nada é apagado: os arquivos só mudam de pasta. Rode primeiro sem --aplicar para ver o que seria
movido. Pode ser rodado de novo depois de cada bateria nova: só move o que ainda está fora do lugar.

Uso (na pasta do projeto):
    python separar_resultados.py              # mostra o que faria
    python separar_resultados.py --aplicar    # move de fato
"""
import argparse
import os
import re
import shutil

SEEDS_DESENVOLVIMENTO = {1993, 1994, 1995, 1996}
# execucoes/ fica de fora: guarda os marcadores .ok que o rodar_experimentos.sh usa para retomar.
SUBPASTAS = ["results", "logs", "checkpoints", "figures_cil"]
PADRAO_SEED = re.compile(r"seed(\d{4})")


def destino_para(nome_arquivo):
    """'desenvolvimento', 'teste' ou None (arquivo sem seed no nome, ex.: resumo.txt)."""
    m = PADRAO_SEED.search(nome_arquivo)
    if not m:
        return None
    return "desenvolvimento" if int(m.group(1)) in SEEDS_DESENVOLVIMENTO else "teste"


def mover(origem, destino, aplicar):
    if os.path.exists(destino):
        print(f"  [JÁ EXISTE, ignorado] {destino}")
        return 0
    print(f"  {origem}  ->  {destino}")
    if aplicar:
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        shutil.move(origem, destino)
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aplicar", action="store_true", help="move os arquivos (sem isso, só mostra)")
    ap.add_argument("--raiz", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados"))
    args = ap.parse_args()
    raiz = args.raiz
    if not os.path.isdir(raiz):
        raise SystemExit(f"Pasta {raiz} não encontrada.")

    n = {"desenvolvimento": 0, "teste": 0}
    sem_seed = []
    for sub in SUBPASTAS:
        pasta = os.path.join(raiz, sub)
        if not os.path.isdir(pasta):
            continue
        print(f"\n[{sub}]")
        for nome in sorted(os.listdir(pasta)):
            grupo = destino_para(nome)
            if grupo is None:
                sem_seed.append(os.path.join(sub, nome))
                continue
            n[grupo] += mover(os.path.join(pasta, nome), os.path.join(raiz, grupo, sub, nome), args.aplicar)

    comp = os.path.join(raiz, "comparacao")
    if os.path.isdir(comp):
        print("\n[comparacao] (mistura seeds de desenvolvimento e teste -> desenvolvimento)")
        for nome in sorted(os.listdir(comp)):
            n["desenvolvimento"] += mover(os.path.join(comp, nome),
                                          os.path.join(raiz, "desenvolvimento", "comparacao", nome), args.aplicar)

    print(f"\nResumo: {n['desenvolvimento']} item(ns) para desenvolvimento, {n['teste']} para teste.")
    if sem_seed:
        print("Sem seed no nome (ficaram onde estão): " + ", ".join(sem_seed))
    if not args.aplicar:
        print("Nada foi movido. Para mover de fato: python separar_resultados.py --aplicar")


if __name__ == "__main__":
    main()
