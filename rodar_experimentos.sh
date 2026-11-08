#!/usr/bin/env bash
# ============================================================================
# Roda a LINHA DE BASE e o EXPERIMENTO da Nova Estratégia 2, em QUATRO BATERIAS, uma por seed:
#   bateria 1 -> seed 1993     bateria 3 -> seed 1995
#   bateria 2 -> seed 1994     bateria 4 -> seed 1996
# Cada bateria = 3 execuções (linha de base, experimento e gauss, todas da mesma seed).
#
#   config_baseline.json    -> pseudo_negativos.modo = "nenhum"  (sem pseudo-negativos)
#   config_experimento.json -> pseudo_negativos.modo = "is"      (pseudo-negativos escolhidos pela margem)
#   config_gauss.json       -> pseudo_negativos.modo = "gauss"   (pseudo-negativos sorteados às cegas;
#                              o replay gaussiano já conhecido na literatura)
#
# Os dois configs são idênticos no resto. Para cada seed, roda primeiro a linha de base e
# depois o experimento, para que mesmo uma execução interrompida deixe pares comparáveis.
#
# Uso (na pasta do projeto, ou de qualquer lugar):
#   bash rodar_experimentos.sh --bateria 1   # seed 1993: linha de base + experimento
#   bash rodar_experimentos.sh --bateria 2   # seed 1994
#   bash rodar_experimentos.sh --bateria 3   # seed 1995
#   bash rodar_experimentos.sh --bateria 4   # seed 1996
#   bash rodar_experimentos.sh --bateria 1 2 # várias baterias em sequência
#   bash rodar_experimentos.sh --teste       # teste rápido: seed 1993 apenas, 2 épocas por tarefa
#   acrescente --refazer para rodar de novo inclusive o que já terminou com sucesso
#
# CORREÇÕES v2 (escala nos logits, classes novas pela média, LVQ desligado, adapter residual):
#   bash rodar_experimentos.sh --v2 --teste      # teste rápido do conjunto v2
#   bash rodar_experimentos.sh --v2 --bateria 1  # seed 1993: ncm_v2 (minutos) + linha de base,
#                                                #   experimento e gauss v2
# Os resultados v2 ficam na mesma pasta resultados/, com "_v2" no nome, e são comparados à parte.
#
# Tudo é gravado DENTRO do projeto, em CIL_e_GFlowNet/resultados/ (logs, results, figures_cil,
# checkpoints, execucoes, comparacao). A comparação no fim de cada bateria usa todas as seeds que
# já tenham as duas execuções: após a bateria 1, só 1993; após a 2, 1993 e 1994; e assim por diante.
#
# Retomada: cada execução bem-sucedida deixa um marcador em resultados/execucoes/*.ok. Se o script
# for interrompido (queda de energia, Ctrl+C), basta chamá-lo de novo: o que já terminou é pulado.
#
# Windows: rode no Git Bash ou no WSL. Para escolher o interpretador:
#   PYTHON=python3 bash rodar_experimentos.sh
# ============================================================================
set -u

PASTA_PROJETO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PASTA_PROJETO" || exit 1

SCRIPT="cil_nova_estrategia_2.py"
# Ordem: os dois braços originais primeiro, depois o terceiro. Execuções já concluídas (marcador .ok)
# são puladas, então baterias rodadas antes do terceiro braço existir só executam o "gauss".
CONFIGS=("config_baseline.json" "config_experimento.json" "config_gauss.json")
# [Correções v2] referência sem treino primeiro (leva minutos), depois os três braços corrigidos
CONFIGS_V2=("config_ncm_v2.json" "config_baseline_v2.json" "config_experimento_v2.json" "config_gauss_v2.json")
# bateria N -> seed SEEDS_POR_BATERIA[N-1]
SEEDS_POR_BATERIA=(1993 1994 1995 1996)
EPOCAS_TESTE=2

MODO_TESTE=0
REFAZER=0
BATERIA=""
while [ $# -gt 0 ]; do
    case "$1" in
        --teste)   MODO_TESTE=1 ;;
        --v2)      CONFIGS=("${CONFIGS_V2[@]}") ;;
        --refazer) REFAZER=1 ;;
        --bateria)
            # aceita uma ou mais baterias: --bateria 1   ou   --bateria 1 2
            while [ $# -gt 1 ] && [[ "$2" =~ ^[0-9]+$ ]]; do BATERIA="$BATERIA $2"; shift; done ;;
        --bateria=*) BATERIA="${1#--bateria=}"; BATERIA="${BATERIA//,/ }" ;;
        -h|--help) sed -n '2,37p' "$0"; exit 0 ;;
        *) echo "Opção desconhecida: $1 (use --bateria 1 a 4, --teste, --v2, --refazer ou --help)"; exit 2 ;;
    esac
    shift
done

if [ "$MODO_TESTE" -eq 1 ]; then
    SEEDS=(1993)
elif [ -n "${BATERIA// /}" ]; then
    SEEDS=()
    for b in $BATERIA; do
        if [ "$b" -ge 1 ] && [ "$b" -le ${#SEEDS_POR_BATERIA[@]} ]; then
            SEEDS+=("${SEEDS_POR_BATERIA[$((b - 1))]}")
        else
            echo "Bateria inválida: $b (use 1 a ${#SEEDS_POR_BATERIA[@]})"; exit 2
        fi
    done
    BATERIA="${BATERIA# }"
else
    echo "Escolha a bateria (uma por seed):"
    for i in "${!SEEDS_POR_BATERIA[@]}"; do
        echo "  bash $0 --bateria $((i + 1))    # seed ${SEEDS_POR_BATERIA[$i]}"
    done
    echo "  bash $0 --teste        # teste rápido (seed 1993, $EPOCAS_TESTE épocas)"
    exit 2
fi

# --- interpretador Python ----------------------------------------------------
if [ -z "${PYTHON:-}" ]; then
    if command -v python >/dev/null 2>&1 && python -c "import sys" >/dev/null 2>&1; then
        PYTHON=python
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON=python3
    else
        echo "ERRO: Python não encontrado. Defina a variável PYTHON, ex.: PYTHON=/caminho/python bash $0"
        exit 1
    fi
fi

# --- verificações antes de começar ---------------------------------------------
for f in "$SCRIPT" pseudo_negativos_cil.py metricas_vazamento.py comparar_resultados.py "${CONFIGS[@]}"; do
    [ -f "$f" ] || { echo "ERRO: arquivo $f não encontrado em $PASTA_PROJETO"; exit 1; }
done
"$PYTHON" -c "import torch, torchvision, sklearn, matplotlib, numpy" 2>/dev/null || {
    echo "ERRO: faltam bibliotecas no Python '$PYTHON'. Instale: pip install torch torchvision scikit-learn matplotlib numpy"
    exit 1
}
GPU=$("$PYTHON" -c "import torch; print('sim' if torch.cuda.is_available() else 'NÃO')")
echo "Python: $("$PYTHON" -c 'import sys; print(sys.executable)') | GPU disponível: $GPU"
[ "$GPU" = "sim" ] || echo "AVISO: sem GPU o treino completo pode levar muitas horas."

PASTA_RESULTADOS="$PASTA_PROJETO/resultados"
PASTA_SAIDA="$PASTA_RESULTADOS/execucoes"
mkdir -p "$PASTA_SAIDA"

# --- modo teste: configs temporários com poucas épocas -------------------------
if [ "$MODO_TESTE" -eq 1 ]; then
    PASTA_TESTE="$PASTA_PROJETO/_configs_teste"
    mkdir -p "$PASTA_TESTE"
    NOVOS=()
    for cfg in "${CONFIGS[@]}"; do
        destino="$PASTA_TESTE/${cfg%.json}_teste.json"
        "$PYTHON" - "$cfg" "$destino" "$EPOCAS_TESTE" <<'EOF'
import json, sys
origem, destino, epocas = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = json.load(open(origem, encoding="utf-8"))
if cfg["treino"].get("epochs_por_tarefa", 1) > 0:      # config_ncm_v2.json continua sem treino
    cfg["treino"]["epochs_por_tarefa"] = epocas
# os caminhos do dataset são resolvidos a partir da pasta do SCRIPT, então não mudam aqui
json.dump(cfg, open(destino, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
EOF
        NOVOS+=("$destino")
    done
    CONFIGS=("${NOVOS[@]}")
    echo "MODO TESTE: seed ${SEEDS[0]}, $EPOCAS_TESTE épocas por tarefa."
fi

# --- execuções -------------------------------------------------------------------
TOTAL=$(( ${#SEEDS[@]} * ${#CONFIGS[@]} ))
if [ "$MODO_TESTE" -eq 0 ]; then
    echo "BATERIA $BATERIA: seeds ${SEEDS[*]} ($TOTAL execuções). Resultados em $PASTA_RESULTADOS"
fi
N=0
RESUMO=()
FALHAS=0
INICIO_GERAL=$(date +%s)

for seed in "${SEEDS[@]}"; do
    for cfg in "${CONFIGS[@]}"; do
        N=$((N + 1))
        nome="$(basename "${cfg%.json}")_seed${seed}"
        saida="$PASTA_SAIDA/${nome}.txt"
        marcador="$PASTA_SAIDA/${nome}.ok"

        if [ "$REFAZER" -eq 0 ] && [ -f "$marcador" ]; then
            echo "[$N/$TOTAL] $nome: já concluído antes, pulando (use --refazer para rodar de novo)."
            RESUMO+=("$nome | pulado (já concluído)")
            continue
        fi

        echo "============================================================"
        echo "[$N/$TOTAL] $nome  —  início $(date '+%d/%m/%Y %H:%M:%S')"
        echo "  saída completa em: $saida"
        ini=$(date +%s)
        # logs/, results/, figures_cil/, checkpoints/ vão para resultados/ dentro do projeto
        "$PYTHON" "$SCRIPT" --config "$cfg" --seed "$seed" > "$saida" 2>&1
        status=$?
        dur=$(( $(date +%s) - ini ))
        dur_txt="$((dur / 3600))h$(( (dur % 3600) / 60 ))min"

        if [ $status -eq 0 ]; then
            touch "$marcador"
            aac=$(grep -o "AAC ([0-9]* tarefas): [0-9.]*%" "$saida" | tail -1)
            vaz=$(grep -A2 "MÉTRICAS DE VAZAMENTO — final" "$saida" | grep -o "Vazamento (Lambda) [. ]*[0-9.]*%" | grep -o "[0-9.]*%$")
            echo "  OK em $dur_txt | ${aac:-AAC não encontrado} | vazamento final: ${vaz:-?}"
            RESUMO+=("$nome | OK | $dur_txt | ${aac:-?} | vazamento ${vaz:-?}")
        else
            FALHAS=$((FALHAS + 1))
            echo "  FALHOU (código $status) em $dur_txt. Últimas linhas:"
            tail -n 15 "$saida" | sed 's/^/    /'
            RESUMO+=("$nome | FALHOU (código $status) | $dur_txt | ver $saida")
        fi
    done
done

# --- resumo ----------------------------------------------------------------------
dur=$(( $(date +%s) - INICIO_GERAL ))
echo "============================================================"
echo "RESUMO${BATERIA:+ DA BATERIA $BATERIA} ($((dur / 3600))h$(( (dur % 3600) / 60 ))min no total)"
for linha in "${RESUMO[@]}"; do echo "  $linha"; done
echo "Resultados detalhados em: $PASTA_RESULTADOS (results, figures_cil, checkpoints, logs)"
{
    echo "Execução de $(date '+%d/%m/%Y %H:%M:%S')${BATERIA:+ — bateria $BATERIA}"
    for linha in "${RESUMO[@]}"; do echo "$linha"; done
} >> "$PASTA_SAIDA/resumo.txt"

# --- gráficos comparativos linha de base x experimento ------------------------------
# Usa todas as seeds que tenham as DUAS execuções concluídas (mesmo que alguma tenha falhado).
echo "============================================================"
echo "Gerando gráficos comparativos (linha de base x experimento)..."
"$PYTHON" comparar_resultados.py --resultados "$PASTA_RESULTADOS/results" --saida "$PASTA_RESULTADOS/comparacao" \
    || echo "AVISO: a comparação falhou; rode à mão: $PYTHON comparar_resultados.py"

[ $FALHAS -eq 0 ] || { echo "$FALHAS execução(ões) falharam."; exit 1; }
