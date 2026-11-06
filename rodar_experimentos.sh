#!/usr/bin/env bash
# ============================================================================
# Roda a LINHA DE BASE e o EXPERIMENTO da Nova Estratégia 2 nas 4 seeds planejadas.
#
#   config_baseline.json    -> pseudo_negativos.modo = "nenhum"  (sem pseudo-negativos)
#   config_experimento.json -> pseudo_negativos.modo = "is"      (com pseudo-negativos)
#
# Os dois configs são idênticos no resto. Para cada seed, roda primeiro a linha de base e
# depois o experimento, para que mesmo uma execução interrompida deixe pares comparáveis.
#
# Uso (na pasta do projeto, ou de qualquer lugar):
#   bash rodar_experimentos.sh            # as 8 execuções (2 configs x 4 seeds), 100 épocas
#   bash rodar_experimentos.sh --teste    # teste rápido: seed 1993 apenas, 2 épocas por tarefa
#   bash rodar_experimentos.sh --refazer  # roda de novo inclusive o que já terminou com sucesso
#
# Retomada: cada execução bem-sucedida deixa um marcador em ../execucoes/*.ok. Se o script
# for interrompido (queda de energia, Ctrl+C), basta chamá-lo de novo: o que já terminou é pulado.
#
# Windows: rode no Git Bash ou no WSL. Para escolher o interpretador:
#   PYTHON=python3 bash rodar_experimentos.sh
# ============================================================================
set -u

PASTA_PROJETO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PASTA_PROJETO" || exit 1

SCRIPT="cil_nova_estrategia_2.py"
CONFIGS=("config_baseline.json" "config_experimento.json")
SEEDS=(1993 1994 1995 1996)
EPOCAS_TESTE=2

MODO_TESTE=0
REFAZER=0
for arg in "$@"; do
    case "$arg" in
        --teste)   MODO_TESTE=1 ;;
        --refazer) REFAZER=1 ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "Opção desconhecida: $arg (use --teste, --refazer ou --help)"; exit 2 ;;
    esac
done

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

PASTA_SAIDA="$(cd .. && pwd)/execucoes"
mkdir -p "$PASTA_SAIDA"

# --- modo teste: configs temporários com poucas épocas -------------------------
if [ "$MODO_TESTE" -eq 1 ]; then
    SEEDS=(1993)
    PASTA_TESTE="$PASTA_PROJETO/_configs_teste"
    mkdir -p "$PASTA_TESTE"
    NOVOS=()
    for cfg in "${CONFIGS[@]}"; do
        destino="$PASTA_TESTE/${cfg%.json}_teste.json"
        "$PYTHON" - "$cfg" "$destino" "$EPOCAS_TESTE" <<'EOF'
import json, sys
origem, destino, epocas = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = json.load(open(origem, encoding="utf-8"))
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
        # a pasta de trabalho é a do projeto: logs/, results/, figures_cil/, checkpoints/ vão para ../
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
echo "RESUMO ($((dur / 3600))h$(( (dur % 3600) / 60 ))min no total)"
for linha in "${RESUMO[@]}"; do echo "  $linha"; done
echo "Resultados detalhados (JSON, figuras, checkpoints) em: $(cd .. && pwd)/results, figures_cil, checkpoints"
{
    echo "Execução de $(date '+%d/%m/%Y %H:%M:%S')"
    for linha in "${RESUMO[@]}"; do echo "$linha"; done
} >> "$PASTA_SAIDA/resumo.txt"

# --- gráficos comparativos linha de base x experimento ------------------------------
# Usa todas as seeds que tenham as DUAS execuções concluídas (mesmo que alguma tenha falhado).
echo "============================================================"
echo "Gerando gráficos comparativos (linha de base x experimento)..."
"$PYTHON" comparar_resultados.py --resultados ../results --saida ../comparacao \
    || echo "AVISO: a comparação falhou; rode à mão: $PYTHON comparar_resultados.py"

[ $FALHAS -eq 0 ] || { echo "$FALHAS execução(ões) falharam."; exit 1; }
