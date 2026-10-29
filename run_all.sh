#!/usr/bin/env bash
###############################################################################
# run_all.sh - TEC508/PGCC011 - Paralelismo com Memória Distribuída (MPI)
# Game of Life: serial (gol.c) e MPI por colunas (gol_mpi.c, com linhas RANK)
#
# Cluster UMBU: rode NO FRONTEND (umbu), em background (como no slide do cluster).
# Os nós são lidos do arquivo hosts_nodes (um nó por linha; "nodeX slots=N" também vale):
#   nohup ./run_all.sh > run_all.out 2>&1 &
#   (outro arquivo: NODES_FILE=meus_nos ./run_all.sh ; ou NODES="node1 node2 ..." )
#   tail -f run_all.out
# Nos nós rodam somente serial, MPI e perf. As tabelas (etapa "tables") são
# geradas no frontend a partir de resultados/raw (o /home é compartilhado).
# Regra: 1 processo MPI por nó  =>  P <= nº de nós (nós inacessíveis são removidos).
#
# ETAPAS (ordem do enunciado): env compile serial strong weak mpip perf tables
#   ./run_all.sh tables   -> só regera as tabelas (usa resultados/parametros.txt)
#
# Repetições: REPS (padrão 3) vale SÓ para strong e weak scaling (média e desvio
# do Tmax). Serial, mpiP e perf rodam 1 vez (SERIAL_REPS=1).
#
# O serial não recebe argumentos (w, h, max_iter fixos no código): mantenha-os
# iguais a W, H e ITER (padrão 500 500 50000).
# A corretude NÃO é verificada por este script (feita à parte, via bitmaps
# periódicos das matrizes para P=1,2,4).
###############################################################################
set -uo pipefail

#============================ CONFIGURAÇÃO ====================================
NODES_FILE="${NODES_FILE:-hosts_nodes}"  # arquivo com os nós (1 por linha; só o nome importa)
NODES="${NODES:-}"                  # alternativa: lista direta, ex.: "node1 node2 ... node12"
if [[ -z "$NODES" && -f "$NODES_FILE" ]]; then
  NODES=$(awk '!/^[[:space:]]*(#|$)/ && !seen[$1]++ {printf "%s ", $1}' "$NODES_FILE")
fi
SKIP_DEAD="${SKIP_DEAD:-1}"         # 1 = remove nós inacessíveis (aviso no log); 0 = aborta
W="${W:-500}"; H="${H:-500}"; ITER="${ITER:-50000}"   # grade W x H e iterações (strong)
WEAK_W1="${WEAK_W1:-$W}"            # weak: largura por processo; w(P)=WEAK_W1*P, h fixo
REPS="${REPS:-3}"                   # repetições SÓ para strong e weak scaling
SERIAL_REPS="${SERIAL_REPS:-1}"     # serial (baseline): 1 execução; use SERIAL_REPS=3 se quiser desvio
RUN_TIMEOUT="${RUN_TIMEOUT:-7200}"  # timeout por execução (detecta deadlock)
LOAD_MAX="${LOAD_MAX:-0.50}"        # loadavg acima disso = nó ocupado por outro grupo
ABORT_ON_BUSY="${ABORT_ON_BUSY:-0}"
MPI_EXTRA="${MPI_EXTRA:---mca routed direct}"  # UMBU; se preciso: --mca oob_tcp_if_include IF --mca btl_tcp_if_include IF
MPIP_LIB="${MPIP_LIB:-/usr/local/lib/libmpiP.so}"  # precisa ser visível nos nós
F_SERIAL="${F_SERIAL-0.0123}"       # fração serial p/ Amdahl (medida no serial: 1,23%); F_SERIAL= (vazio) = estima por Karp-Flatt
pick() { for f in "$@"; do [[ -f $f ]] && { echo "$f"; return; }; done; echo "$1"; }   # 1º arquivo existente
SERIAL_SRC="${SERIAL_SRC:-$(pick gol.c gol_serial.c programa_serial.c)}"
MPI_SRC="${MPI_SRC:-$(pick golmpi.c gol_mpi.c programa_mpi.c)}"
COMPILE_ON_NODE="${COMPILE_ON_NODE:-0}"  # 1 = compila via ssh no 1º nó

SERIAL_BIN=programa_serial; MPI_BIN=programa_mpi
OUT="${OUT:-$PWD/resultados}"
RAW="$OUT/raw"; PERF_DIR="$OUT/perf"; MPIP_DIR="$OUT/mpip"; TAB="$OUT/tabelas"
mkdir -p "$RAW" "$PERF_DIR" "$MPIP_DIR" "$TAB"
HOSTFILE="$OUT/machinefile"; LOG="$OUT/comandos_executados.log"
PERF_EVENTS="cycles,instructions,cache-references,cache-misses,branches,branch-misses"

STEPS=("$@"); [[ ${#STEPS[@]} -eq 0 ]] && STEPS=(all)
want() { [[ " ${STEPS[*]} " == *" all "* || " ${STEPS[*]} " == *" $1 "* ]]; }
log() { echo "[$(date +%F' '%T)] $*" | tee -a "$LOG"; }

#=========================== FUNÇÕES AUXILIARES ===============================
# Define NODE_LIST, FIRST_NODE e P_LIST (padrão: 1,2,4,... e o nº total de nós)
USER_P_LIST="${P_LIST:-}"; USER_MPIP_P="${MPIP_P:-}"
set_lists() {
  NODE_LIST="$*"; NNODES=$#; FIRST_NODE=${1:-}
  if [[ -n "$USER_P_LIST" ]]; then P_LIST="$USER_P_LIST"
  elif (( NNODES > 0 )); then
    P_LIST=$(echo $(awk -v n="$NNODES" 'BEGIN{for(p=1;p<=n;p*=2) s=s p " "; if(p/2!=n) s=s n; print s}'))
  else P_LIST="1 2 4 8 12"; fi
  MAXP=$(tr ' ' '\n' <<< "$P_LIST" | sort -n | tail -1)
  MPIP_P="${USER_MPIP_P:-$MAXP}"
}

# executa um comando no 1º nó (mesmo diretório, via /home compartilhado)
on_node() { ssh -n -o BatchMode=yes "$FIRST_NODE" "cd '$PWD' && $*"; }

# registra a carga dos nós (anotar interferência de outros grupos)
snap() {
  { echo "=== $1 $(date) ==="
    for n in $NODE_LIST; do echo "$n: $(ssh -n -o BatchMode=yes "$n" 'cat /proc/loadavg' 2>&1)"; done; } >> "$OUT/interferencia.log"
}

# mpirun: exatamente 1 processo por nó, sem oversubscribe.
# Sempre com nohup (sem acoplamento ao terminal); quem chama executa em background (&) e espera (wait).
build_mpi_cmd() { MPI_CMD=(nohup mpirun -np "$1" -machinefile "$HOSTFILE" --map-by ppr:1:node --bind-to core $MPI_EXTRA); }

# run_series TAG PROG P w h : REPS execuções (strong/weak); .out (stdout), .time (/usr/bin/time -v do mpirun), .err
run_series() {
  local tag=$1 prog=$2 P=$3 w=$4 h=$5 i base rc
  build_mpi_cmd "$P"
  for i in $(seq 1 "$REPS"); do
    base="$RAW/${tag}_P${P}_W${w}_H${h}_r${i}"
    log "RUN $tag P=$P grade=${w}x${h} iter=$ITER rep=$i"
    # nohup mpirun ... &  (background, sem terminal); wait espera o término para medir/seguir a série
    /usr/bin/time -v -o "$base.time" timeout "$RUN_TIMEOUT" "${MPI_CMD[@]}" "$prog" "$w" "$h" "$ITER" \
        < /dev/null > "$base.out" 2> "$base.err" &
    wait $!; rc=$?; (( rc != 0 )) && log "  AVISO: retorno $rc (veja $base.err)"
  done
}

# have_series P w h : 0 se as REPS execuções já existem em raw/ (com linhas RANK)
have_series() { local i; for i in $(seq 1 "$REPS"); do grep -q '^RANK' "$RAW/mpi_P$1_W$2_H$3_r$i.out" 2>/dev/null || return 1; done; }

# preflight: testa ssh/carga de cada nó (remove os inacessíveis), valida P e gera o machinefile
preflight() {
  [[ -n "$NODES" ]] || { echo "Nenhum nó: crie o arquivo $NODES_FILE (1 nó por linha) ou defina NODES=\"node1 node2 ...\"" >&2; exit 1; }
  local n ld busy=0 p alive=() dead=()
  for n in $NODES; do
    if ld=$(ssh -n -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$n" "cut -d' ' -f1 /proc/loadavg" 2>"$OUT/.ssh_err"); then
      alive+=("$n"); log "preflight: $n ok, loadavg=$ld"
      awk -v l="$ld" -v m="$LOAD_MAX" 'BEGIN{exit !(l>m)}' && { busy=1; log "AVISO: $n ocupado (load $ld). Coordene com outros grupos."; }
    elif grep -qiE "permission denied|host key" "$OUT/.ssh_err"; then
      echo "ERRO: ssh sem senha para $n falhou ($(head -1 "$OUT/.ssh_err")). Com /home compartilhado, uma vez:" >&2
      echo "  ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519; cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys" >&2
      exit 1
    else
      dead+=("$n"); log "AVISO: $n inacessível ($(head -1 "$OUT/.ssh_err"))"
      [[ "$SKIP_DEAD" == 1 ]] || { echo "Abortando (SKIP_DEAD=0)." >&2; exit 1; }
    fi
  done
  (( ${#alive[@]} > 0 )) || { echo "Nenhum nó acessível." >&2; exit 1; }
  set_lists "${alive[@]}"
  for c in gcc mpicc mpirun python3; do command -v "$c" >/dev/null 2>&1 || log "AVISO: '$c' não encontrado no frontend."; done
  if want serial || want strong || want weak; then
    [[ -x /usr/bin/time ]] || { echo "ERRO: /usr/bin/time não existe no frontend (instale o pacote 'time')." >&2; exit 1; }
    on_node "test -x /usr/bin/time" || { echo "ERRO: /usr/bin/time não existe em $FIRST_NODE." >&2; exit 1; }
  fi
  for p in $P_LIST; do
    (( p <= NNODES )) || { echo "ERRO: P=$p > $NNODES nós acessíveis (máximo 1 processo por nó)" >&2; exit 1; }
    (( p <= W )) || { echo "ERRO: P=$p > largura W=$W" >&2; exit 1; }
  done
  (( busy && ABORT_ON_BUSY )) && { echo "Abortando: nós ocupados." >&2; exit 1; }
  : > "$HOSTFILE"; for n in $NODE_LIST; do echo "$n slots=1" >> "$HOSTFILE"; done
  cat > "$OUT/parametros.txt" <<EOF
W=$W; H=$H; ITER=$ITER; REPS=$REPS; SERIAL_REPS=$SERIAL_REPS; WEAK_W1=$WEAK_W1
NODES_USADOS="$NODE_LIST"; NODES_INACESSIVEIS="${dead[*]:-}"
P_LIST="$P_LIST"; MPIP_P=$MPIP_P
EOF
  log "nós usados ($NNODES): $NODE_LIST | P_LIST: $P_LIST"
}

if [[ ${#STEPS[@]} -eq 1 && "${STEPS[0]}" == tables ]]; then
  set_lists   # só tabelas: usa os parâmetros salvos na última execução
  # shellcheck disable=SC1090
  [[ -f "$OUT/parametros.txt" ]] && source "$OUT/parametros.txt"
else preflight; fi

#================================ ETAPAS ======================================

# ---- env (seção 7): hardware, memória, rede, latência -----------------------
if want env; then
  log "### env"
  {
    echo "### frontend"; hostname; date; grep PRETTY_NAME /etc/os-release; df -hT "$HOME"
    echo "### versões"; gcc --version | head -1; mpicc --version | head -1; mpirun --version | head -1
    for n in $NODE_LIST; do
      echo; echo "######## $n"
      ssh -n -o BatchMode=yes "$n" 'grep PRETTY_NAME /etc/os-release; lscpu; free -h; ip -br addr; for i in /sys/class/net/*; do echo "$(basename $i) $(cat $i/speed 2>/dev/null) Mb/s"; done; perf --version 2>&1'
    done
    echo; echo "### latência ICMP de $FIRST_NODE para os demais nós"
    for n in $NODE_LIST; do [[ $n != "$FIRST_NODE" ]] && { echo "-> $n"; on_node "ping -c 10 -q $n | tail -2"; }; done
  } > "$OUT/ambiente.txt" 2>&1
fi

# ---- compile (2.1): comandos do enunciado -----------------------------------
if want compile; then
  log "### compile"
  [[ -f "$SERIAL_SRC" && -f "$MPI_SRC" ]] || { log "ERRO: fonte não encontrado (SERIAL_SRC=$SERIAL_SRC, MPI_SRC=$MPI_SRC); ajuste as variáveis."; exit 1; }
  build() { log "$*"; if [[ "$COMPILE_ON_NODE" == 1 ]]; then on_node "$*"; else bash -c "$*"; fi; }
  build "gcc -O2 -g -o $SERIAL_BIN $SERIAL_SRC"  || exit 1
  build "mpicc -O2 -g -o $MPI_BIN $MPI_SRC"      || exit 1
fi

# ---- serial (2.1.1 / 2.2.2): baseline no nó, com /usr/bin/time -v ------------
if want serial; then
  log "### serial em $FIRST_NODE (grade fixa no código; esperado ${W}x${H}, ${ITER} iterações; SERIAL_REPS=$SERIAL_REPS)"; snap serial
  for i in $(seq 1 "$SERIAL_REPS"); do
    base="$RAW/serial_r$i"; log "RUN serial rep=$i"
    on_node "/usr/bin/time -v -o $base.time ./$SERIAL_BIN > $base.out 2> $base.err"
  done
fi

# ---- strong (2.2, 2.3, 2.5, 2.8): grade fixa, P variável, REPS execuções -----
if want strong; then
  log "### strong: grade ${W}x${H}, P={$P_LIST}, REPS=$REPS"; snap strong
  for P in $P_LIST; do run_series mpi "./$MPI_BIN" "$P" "$W" "$H"; done
fi

# ---- weak (2.4): w(P)=WEAK_W1*P, h fixo, REPS execuções ----------------------
if want weak; then
  log "### weak: w=${WEAK_W1}*P, h=$H, REPS=$REPS"; snap weak
  : > "$OUT/weak_manifest.txt"
  for P in $P_LIST; do
    WW=$(( WEAK_W1 * P )); echo "$P $WW $H" >> "$OUT/weak_manifest.txt"
    if (( WW == W )) && have_series "$P" "$WW" "$H"; then
      log "weak P=$P (${WW}x$H) = mesma grade do strong; reaproveitando as $REPS execuções existentes"
    else run_series mpi "./$MPI_BIN" "$P" "$WW" "$H"; fi
  done
fi

# ---- mpiP (2.6): LD_PRELOAD só nos ranks, 1 execução --------------------------
if want mpip; then
  log "### mpip: P=$MPIP_P (1 execução)"
  if [[ ! -f "$MPIP_LIB" ]]; then log "AVISO: $MPIP_LIB não existe (ajuste MPIP_LIB); etapa pulada."
  else
    snap mpip
    export MPIP="-f $MPIP_DIR -k 2"          # -f: pasta de saída; -k 2: pilha p/ arquivo:linha
    build_mpi_cmd "$MPIP_P"
    log "${MPI_CMD[*]} -x LD_PRELOAD=$MPIP_LIB -x MPIP ./$MPI_BIN $W $H $ITER"
    "${MPI_CMD[@]}" -x LD_PRELOAD="$MPIP_LIB" -x MPIP "./$MPI_BIN" "$W" "$H" "$ITER" < /dev/null > "$MPIP_DIR/stdout.txt" 2> "$MPIP_DIR/stderr.txt" &
    wait $!
    unset MPIP; ls -l "$MPIP_DIR"/*.mpiP 2>&1 | tee -a "$LOG"
  fi
fi

# ---- perf (2.7): serial, mpirun em 1 nó e uma instância de perf por nó (1 execução)
if want perf; then
  log "### perf (1 execução)"
  if ! on_node "perf stat -o /dev/null true" >/dev/null 2>&1; then
    log "AVISO: perf indisponível/sem permissão (perf_event_paranoid). Etapa pulada."
  else
    snap perf
    rm -f "$PERF_DIR"/serial.csv "$PERF_DIR"/mpirun_P*.csv "$PERF_DIR"/rank_P*.csv
    PNP="${PERF_NP:-$(on_node 'nproc' 2>/dev/null || echo 1)}"   # P = núcleos do nó (enunciado 2.7)
    (( PNP > W )) && PNP=$W
    log "perf: serial e 'perf stat mpirun -np $PNP' em $FIRST_NODE (1 nó)"
    on_node "perf stat -x, -o $PERF_DIR/serial.csv -e $PERF_EVENTS ./$SERIAL_BIN >/dev/null 2>&1"
    on_node "perf stat -x, -o $PERF_DIR/mpirun_P${PNP}.csv -e $PERF_EVENTS nohup mpirun -np $PNP --host $FIRST_NODE:$PNP $MPI_EXTRA ./$MPI_BIN $W $H $ITER >/dev/null 2>&1 < /dev/null & wait"
    # multinó: perf dentro de cada nó via wrapper (1 rank por nó => 1 instância por nó)
    cat > "$OUT/perf_wrap.sh" <<'WRAP'
#!/usr/bin/env bash
exec perf stat -x, -o "$PERF_DIR/rank_P${PERF_P}_$(hostname -s)_${OMPI_COMM_WORLD_RANK:-0}.csv" -e "$PERF_EVENTS" "$@"
WRAP
    chmod +x "$OUT/perf_wrap.sh"; export PERF_DIR PERF_EVENTS; export PERF_P="$MAXP"
    build_mpi_cmd "$MAXP"
    "${MPI_CMD[@]}" -x PERF_DIR -x PERF_EVENTS -x PERF_P "$OUT/perf_wrap.sh" "./$MPI_BIN" "$W" "$H" "$ITER" < /dev/null >/dev/null 2>&1 &
    wait $!
  fi
fi

# ---- tables: RODA NO FRONTEND. Gera as tabelas a partir de resultados/raw ----
if want tables; then
  log "### tables (frontend)"
cat > "$OUT/analyze.py" <<'PYEOF'
# -*- coding: utf-8 -*-
import os, re, sys, glob, statistics as st
OUT=os.environ["OUT"]; RAW=OUT+"/raw"; TAB=OUT+"/tabelas"; PERF=OUT+"/perf"; MPIP=OUT+"/mpip"
P_LIST=[int(x) for x in os.environ["P_LIST"].split()]
W=int(os.environ["W"]); H=int(os.environ["H"]); MPIP_P=int(os.environ["MPIP_P"])
F_SERIAL=os.environ.get("F_SERIAL",""); NAN=float("nan")

def mean(x): return sum(x)/len(x) if x else NAN
def sd(x):  return st.stdev(x) if len(x)>1 else 0.0     # entre execuções
def psd(x): return st.pstdev(x) if len(x)>1 else 0.0    # entre ranks
def fmt(x,d=3): return "n/d" if x!=x else f"{x:.{d}f}"
def md(h,rows):
    s="| "+" | ".join(h)+" |\n|"+"|".join(["---"]*len(h))+"|\n"
    return s+"".join("| "+" | ".join(str(c) for c in r)+" |\n" for r in rows)
def save(name,title,body):
    t=f"## {title}\n\n{body}\n"; open(f"{TAB}/{name}.md","w").write(t); print(t)

# ---- leitura dos arquivos brutos ----
def time_of(p):
    d={}
    if not os.path.exists(p): return d
    for l in open(p,errors="ignore"):
        try:
            if "Elapsed (wall clock)" in l:
                v=[float(x) for x in l.strip().rsplit(" ",1)[1].split(":")]; d["wall"]=sum(x*60**i for i,x in enumerate(reversed(v)))
            elif "User time" in l: d["user"]=float(l.rsplit(" ",1)[1])
            elif "System time" in l: d["sys"]=float(l.rsplit(" ",1)[1])
            elif "Maximum resident" in l: d["rss"]=float(l.rsplit(" ",1)[1])/1024
        except Exception: pass
    return d
def parse_mpi(p):
    """Linhas RANK (uma por rank) + 'Total de celulas vivas' do gol_mpi.c."""
    ranks=[];alive=None
    for l in open(p,errors="ignore"):
        m=re.match(r"\s*RANK\s+(\d+)\s+HOST\s+(\S+)\s+TCOMP\s+(\S+)\s+TCOMM\s+(\S+)\s+TTOTAL\s+(\S+)\s+BYTES\s+(\d+)",l)
        if m: ranks.append(dict(rank=int(m[1]),host=m[2],comp=float(m[3]),comm=float(m[4]),tot=float(m[5]),bytes=int(m[6]))); continue
        m=re.match(r"\s*Total de celulas vivas ao final:\s*(\d+)",l)
        if m: alive=int(m[1])
    if not ranks: print(f"(aviso: {os.path.basename(p)} sem linhas RANK; ignorado)"); return None
    return dict(ranks=ranks,alive=alive,tot=max(r["tot"] for r in ranks),
                comp_max=max(r["comp"] for r in ranks),comm_max=max(r["comm"] for r in ranks),
                bytes=sum(r["bytes"] for r in ranks))
def runs(P,w,h):
    out=[]
    for p in sorted(glob.glob(f"{RAW}/mpi_P{P}_W{w}_H{h}_r*.out")):
        d=parse_mpi(p)
        if d: d["time"]=time_of(p[:-4]+".time"); out.append(d)
    return out
def tmax(rs): return [x["tot"] for x in rs]                       # Tmax = maior T_total entre os ranks
def fcomm(rs): return [x["comm_max"]/x["tot"] for x in rs]        # F_comm = Tcomm_max / Ttotal_max
def serial_base():
    rows=[]
    for p in sorted(glob.glob(f"{RAW}/serial_r*.out")):
        t=a=None
        for l in open(p,errors="ignore"):
            m=re.match(r"\s*Tempo interno de execucao:\s*(\S+)",l)
            if m: t=float(m[1])
            m=re.match(r"\s*Total de celulas vivas ao final:\s*(\d+)",l)
            if m: a=int(m[1])
        if t is not None: rows.append((os.path.basename(p),t,time_of(p[:-4]+".time"),a))
    return rows
def TS():
    r=serial_base()
    if not r: sys.exit("sem dados do serial (rode a etapa serial)")
    return mean([x[1] for x in r]), sd([x[1] for x in r])

# ---- 2.1.1 baseline serial ----
def serial():
    rows=serial_base(); ts=[r[1] for r in rows]
    body=md(["execução","tempo interno (s)","wall (time -v)","user","system","Max RSS (MB)"],
        [[n,fmt(t),fmt(tm.get("wall",NAN)),fmt(tm.get("user",NAN)),fmt(tm.get("sys",NAN)),fmt(tm.get("rss",NAN),1)] for n,t,tm,a in rows])
    body+=f"\n**T_serial médio = {mean(ts):.3f} s (desvio {sd(ts):.3f}; {len(ts)} execução(ões))**\n"
    if mean(ts)<200: body+="\n> ⚠ T_serial < 200 s: aumente a grade ou as iterações (e recompile o serial).\n"
    save("01_serial","Baseline serial (2.1.1)",body)

# ---- 2.2 medição de tempo ----
def times():
    rows=[];rows2=[];rows3=[];dump=""
    try: ts_=TS()[0]
    except SystemExit: ts_=NAN
    for P in P_LIST:
        rs=runs(P,W,H)
        if not rs: continue
        for k,name in (("comp","T_comp"),("comm","T_comm"),("tot","T_total")):
            v=lambda f:mean([f([r[k] for r in x["ranks"]]) for x in rs])
            rows.append([P,name,fmt(v(max),4),fmt(v(min),4),fmt(v(mean),4),fmt(v(psd),4)])
        tl=tmax(rs); tm=mean(tl); g=lambda k:mean([x["time"].get(k,NAN) for x in rs]); s_=lambda k:sd([x["time"].get(k,NAN) for x in rs])
        rows2.append([P,fmt(tm),fmt(sd(tl)),fmt(ts_/tm,2),fmt(ts_/tm/P),fmt(g("wall")),fmt(s_("wall")),fmt(g("user"),2),fmt(s_("user"),2),
                      fmt(g("sys"),2),fmt(s_("sys"),2),fmt(g("rss"),1),fmt(s_("rss"),1),fmt(g("wall")-tm)])
        for i,x in enumerate(rs,1):
            tt=x["time"]; rows3.append([P,i,fmt(x["tot"]),fmt(tt.get("wall",NAN)),fmt(tt.get("user",NAN),2),fmt(tt.get("sys",NAN),2),fmt(tt.get("rss",NAN),1)])
        dump+=f"\n**P={P}, execução 1**\n\n"+md(["rank","host","T_comp","T_comm","T_total","bytes enviados"],
            [[r["rank"],r["host"],fmt(r["comp"],4),fmt(r["comm"],4),fmt(r["tot"],4),r["bytes"]] for r in sorted(rs[0]["ranks"],key=lambda z:z["rank"])])
    b="### MPI_Wtime (média das execuções; desvio entre ranks)\n\n"+md(["P","métrica","Tmax (s)","Tmin (s)","Tmédia (s)","desvio"],rows)
    b+=("\nT_comp = inicialização + evolve + contagem local (fora de MPI); T_comm = halo (Sendrecv) + Reduce + barreira final (dentro de MPI). "
        "A barreira final conta como comunicação: a espera pelo rank mais lento aparece em T_comm dos demais.\n")
    b+="\n### MPI_Wtime vs /usr/bin/time -v (sobre o mpirun): média e desvio entre as execuções\n\n"+md(
        ["P","Tmax Wtime (s)","desvio","Speedup (Tmax)","Eficiência (Tmax)","wall (s)","desvio","user (s)","desvio","system (s)","desvio","Max RSS (MB)","desvio","wall − Tmax (s)"],rows2)
    b+="\nSpeedup e eficiência calculados com o Tmax (MPI_Wtime) e o T_serial médio; o wall do time -v inclui o runtime e não serve de base.\n"
    b+="\n### Todas as execuções lado a lado (Tmax Wtime vs time -v)\n\n"+md(["P","exec","Tmax Wtime (s)","wall (s)","user (s)","system (s)","Max RSS (MB)"],rows3)
    b+=("\n> **Limitação do /usr/bin/time:** mede o processo launcher (mpirun/orterun, que roda no frontend), não cada rank. O wall inclui "
        "início/fim do runtime (ssh/orted), user/system são do launcher e Max RSS não representa a memória dos ranks remotos. "
        "O tempo paralelo confiável é Tmax por MPI_Wtime: o rank mais lento limita o paralelo (a seção termina na barreira final); "
        "Tmédia próximo de Tmax indica boa distribuição de carga.\n")
    save("03_tempos","Medição de tempo (2.2)",b+"\n### Valores por rank\n"+dump)

# ---- 2.3 strong scaling ----
def strong():
    ts,tsd=TS(); data={}
    for P in P_LIST:
        rs=runs(P,W,H)
        if rs: t=tmax(rs); data[P]=(mean(t),sd(t))
    f,orig=0.0,"n/d"
    if F_SERIAL: f,orig=float(F_SERIAL),"F_SERIAL"
    else:
        pts=[(1-1/P, data[P][0]/ts-1/P) for P in sorted(data) if P>1]   # 1/S - 1/P = f*(1-1/P)
        if pts:
            f=min(max(sum(x*y for x,y in pts)/sum(x*x for x,_ in pts),0),1)
            orig=f"ajuste por mínimos quadrados sobre todos os P>1 ({len(pts)} pontos)"+(" - com 1 ponto a curva passa por ele" if len(pts)==1 else "")
    rows=[];csv="P,Tmax_medio,desvio,speedup,eficiencia,amdahl\n"
    for P in sorted(data):
        m,s=data[P]; S=ts/m; A=1/(f+(1-f)/P); kf=(1/S-1/P)/(1-1/P) if P>1 else NAN
        rows.append([P,fmt(m),fmt(s),fmt(S,2),fmt(S/P),P,fmt(A,2),fmt(kf,4)])
        csv+=f"{P},{m:.6f},{s:.6f},{S:.4f},{S/P:.4f},{A:.4f}\n"
    open(OUT+"/dados_strong_scaling.csv","w").write(csv)
    b=f"T_serial = {ts:.3f} s (desvio {tsd:.3f}); grade {W}x{H}\n\n"+md(["P","Tmax médio (s)","desvio","Speedup","Eficiência","Ideal","Amdahl","Karp-Flatt e"],rows)
    b+=f"\nFração serial f = {f:.4f} ({orig}); Amdahl: S=1/(f+(1-f)/P).\n"
    b+=("\n**Discussão:** onde a curva se afasta do ideal e por quê. Karp-Flatt *e* constante ⇒ fração serial; *e* crescente com P ⇒ "
        "comunicação (halo/rede) ou desbalanceamento (ver tabelas de comunicação e balanceamento).\n")
    save("04_strong","Strong scaling (2.3)",b)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        Ps=sorted(data); plt.figure(figsize=(6,4))
        plt.plot(Ps,[ts/data[p][0] for p in Ps],"o-",label="medido"); plt.plot(Ps,Ps,"k--",label="ideal")
        plt.plot(Ps,[1/(f+(1-f)/p) for p in Ps],"s:",label=f"Amdahl (f={f:.3f})")
        plt.xlabel("P (nós)"); plt.ylabel("Speedup"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(OUT+"/speedup_strong.png",dpi=150)
    except Exception as e: print("(gráfico não gerado:",e,")")

# ---- 2.4 weak scaling ----
def weak():
    rows=[]
    man=[[int(v) for v in l.split()] for l in open(OUT+"/weak_manifest.txt")]
    r1=[runs(P,w,h) for P,w,h in man if P==1]
    base=mean(tmax(r1[0])) if r1 and r1[0] else None      # T(1) real; sem P=1 a eficiência fica n/d
    for P,w,h in man:
        rs=runs(P,w,h)
        if not rs: continue
        t=tmax(rs); m=mean(t)
        rows.append([P,f"{w}x{h}",f"{w*h:,}",fmt(m),fmt(sd(t)),fmt(base/m) if base else "n/d",fmt(100*mean(fcomm(rs)),2),int(mean([x["bytes"] for x in rs]))])
    b=md(["P","grade (w x h)","N_total (células)","Tmax médio (s)","desvio","Eficiência fraca T(1)/T(P)","F_comm (%)","bytes enviados (total)"],rows)
    b+=("\n**Discussão:** tempo ≈ constante ⇒ boa escalabilidade fraca (Gustafson). Cada rank mantém ~w/P colunas e troca sempre 2 colunas de h células: "
        "o volume por rank é constante, mas o total enviado cresce com P e a latência/sincronização do anel pode crescer com P.\n")
    save("05_weak","Weak scaling (2.4)",b)

# ---- 2.5 comunicação vs computação ----
def comm():
    rows=[]
    for P in P_LIST:
        rs=runs(P,W,H)
        if not rs: continue
        rows.append([P,fmt(mean([mean([r["comp"] for r in x["ranks"]]) for x in rs]),4),fmt(mean([mean([r["comm"] for r in x["ranks"]]) for x in rs]),4),
                     fmt(100*mean(fcomm(rs)),2),int(mean([x["bytes"] for x in rs]))])
    save("06_comunicacao","Comunicação vs computação (2.5)",
         md(["P","T_comp médio (s)","T_comm médio (s)","F_comm (%) = Tcomm_max/Ttotal_max","bytes enviados (total)"],rows)
         +"\n**Discussão:** F_comm crescente com P ⇒ overhead relativo maior ⇒ saturação do strong scaling.\n")

# ---- 2.6 mpiP ----
def sections(path):
    s={};cur=None
    for l in open(path,errors="ignore"):
        if l.startswith("@---"): cur=l.strip("@- \n"); s[cur]=[]
        elif cur and not l.startswith("---") and l.strip(): s[cur].append(l.split())
    return s
def sec(s,pre):
    for k,v in s.items():
        if k.startswith(pre): return v
    return []
def mpip():
    fs=sorted(glob.glob(MPIP+"/*.mpiP"),key=os.path.getmtime)
    if not fs: print("sem relatório mpiP"); return
    s=sections(fs[-1]); b=f"Relatório: `{os.path.basename(fs[-1])}` (P={MPIP_P})\n\n"
    sites={int(t[0]):(f"{t[2]}:{t[3]}",t[4]) for t in sec(s,"Callsites") if len(t)>=6 and t[0].isdigit()}
    app=pct=NAN
    for t in sec(s,"MPI Time"):
        if t[0]=="*": app,pct=float(t[1]),float(t[3])
    agg=[(t[0],int(t[1]),float(t[2]),t[3],t[4]) for t in sec(s,"Aggregate Time") if len(t)>=5 and t[1].isdigit()]
    sent={(t[0],int(t[1])):int(float(t[3])) for t in sec(s,"Aggregate Sent Message Size") if len(t)>=6 and t[1].isdigit()}
    cnt={(t[0],int(t[1])):int(t[3]) for t in sec(s,"Callsite Time statistics") if len(t)>=9 and t[2]=="*" and t[1].isdigit()}
    fn={}
    for c,_,tm,_,_ in agg: fn[c]=fn.get(c,0)+tm
    tot=sum(fn.values()) or 1
    b+="### Funções MPI por tempo total\n\n"+md(["função","tempo total (ms)","% do tempo MPI"],[["MPI_"+c,fmt(v,1),fmt(100*v/tot,1)] for c,v in sorted(fn.items(),key=lambda z:-z[1])])
    b+="\n### 3 callsites mais caros\n\n"+md(["#","chamada","site","arquivo:linha","função","nº chamadas","tempo total (ms)","App%","MPI%","bytes enviados"],
        [[i+1,"MPI_"+c,site,sites.get(site,("?","?"))[0],sites.get(site,("?","?"))[1],cnt.get((c,site),"n/d"),fmt(tm,1),a,m,sent.get((c,site),0)]
         for i,(c,site,tm,a,m) in enumerate(sorted(agg,key=lambda z:-z[2])[:3])])
    rs=runs(MPIP_P,W,H); man=NAN; tm_=NAN
    if rs:
        man=100*mean([sum(r["comm"] for r in x["ranks"])/sum(r["tot"] for r in x["ranks"]) for x in rs]); tm_=mean(tmax(rs))
    b+="\n### mpiP vs instrumentação manual\n\n"+md(["fonte","tempo de referência (s)","% do tempo em MPI"],
        [["mpiP (MPI_Init→Finalize, média por rank)",fmt(app/MPIP_P if app==app else NAN),fmt(pct,2)],
         ["manual (soma Tcomm / soma Ttotal dos ranks, barreira→barreira)",fmt(tm_),fmt(man,2)]])
    b+=("\n**Discussão:** mpiP mede de MPI_Init a MPI_Finalize e cobre todas as chamadas sem alterar o código (com overhead de interposição); "
        "a medição manual cobre só a seção paralela, separa fases do algoritmo e inclui o custo do MPI_Wtime.\n")
    save("07_mpip","Profiling com mpiP (2.6)",b)

# ---- 2.7 perf ----
EV={"cpu-cycles":"cycles","cycles":"cycles","instructions":"instructions","cache-references":"cache-references","cache-misses":"cache-misses",
    "branches":"branches","branch-instructions":"branches","branch-misses":"branch-misses"}
def pfile(p):
    d={}
    for l in open(p,errors="ignore"):
        t=l.strip().split(",")
        if len(t)<3: continue
        try: v=float(t[0])
        except ValueError: continue
        e=EV.get(t[2].split(":")[0])
        if e: d[e]=d.get(e,0)+v
    return d
def prow(lab,d):
    g=lambda k:d.get(k,NAN); r=lambda a,b:100*g(a)/g(b) if g(b) else NAN
    return [lab,f"{g('cycles'):.3e}",f"{g('instructions'):.3e}",fmt(g("instructions")/g("cycles") if g("cycles") else NAN,2),
            f"{g('cache-references'):.3e}",f"{g('cache-misses'):.3e}",fmt(r("cache-misses","cache-references"),2),
            f"{g('branches'):.3e}",f"{g('branch-misses'):.3e}",fmt(r("branch-misses","branches"),2)]
def perf():
    H_=["execução","cycles","instructions","IPC","cache-refs","cache-misses","miss %","branches","branch-misses","miss %"]; rows=[]
    if os.path.exists(PERF+"/serial.csv"): rows.append(prow("serial (1 nó)",pfile(PERF+"/serial.csv")))
    for f in sorted(glob.glob(PERF+"/mpirun_P*.csv"),key=lambda z:int(re.search(r"mpirun_P(\d+)",z)[1])):
        np_=re.search(r"mpirun_P(\d+)",f)[1]; rows.append(prow("perf stat mpirun P="+np_+" (1 nó)",pfile(f)))
    b="### perf stat em um nó\n\n"+md(H_,rows); byh={}
    for p in glob.glob(PERF+"/rank_P*_*.csv"):
        m=re.match(r".*/rank_P(\d+)_(.+)_(\d+)\.csv",p)
        if m:
            a=byh.setdefault((int(m[1]),m[2]),{})
            for k,v in pfile(p).items(): a[k]=a.get(k,0)+v
    if byh: b+="\n### perf stat por nó (uma instância de perf em cada nó)\n\n"+md(H_,[prow(f"P={P} nó {h}",d) for (P,h),d in sorted(byh.items())])
    b+=("\n> **Limitação:** o perf mede apenas o nó em que roda e os filhos locais; `perf stat mpirun` no frontend não enxerga os ranks remotos e não existe "
        "relógio/contador de hardware único e coerente entre nós. Por isso as métricas são coletadas por nó (uma instância de perf por nó) e comparadas nó a nó.\n"
        "\n**Discussão:** compare IPC e taxa de cache-miss do MPI com o serial.\n")
    save("08_perf","perf stat por nó (2.7)",b)

# ---- 2.8 balanceamento ----
def balance():
    ts,_=TS(); rows=[];dump="";TH=1.05
    for P in P_LIST:
        rs=runs(P,W,H)
        if not rs: continue
        by={}
        for x in rs:
            for r in x["ranks"]: by.setdefault(r["rank"],[]).append(r["comp"])
        ks=sorted(by); c=[mean(by[k]) for k in ks]; t=mean(tmax(rs)); imb=max(c)/mean(c); tb=t-(max(c)-mean(c))
        rows.append([P,fmt(max(c),4),fmt(min(c),4),fmt(mean(c),4),fmt(psd(c),4),fmt(imb),fmt(100*mean(c)/max(c),1),fmt(ts/t,2),fmt(ts/tb,2),"balanceado" if imb<=TH else "DESBALANCEADO"])
        dump+=f"\n**P={P}**\n\n"+md(["rank","T_comp médio (s)","T_comp/Tmédia"],[[k,fmt(v,4),fmt(v/mean(c))] for k,v in zip(ks,c)])
    b=md(["P","Tcomp max","Tcomp min","Tcomp média","desvio","Tmax/Tmédia","efic. balanceamento (%)","Speedup medido","Speedup se balanceado","classificação"],rows)
    b+=(f"\nCritério: balanceado se Tmax/Tmédia ≤ {TH}. 'Speedup se balanceado' remove de Tmax o excesso (Tcomp_max − Tcomp_média). "
        "No Game of Life por colunas a carga por célula é uniforme; a diferença vem só das colunas extras dos primeiros `w mod P` ranks (no máximo 1 coluna) e de ruído do SO/rede.\n")
    save("09_balanceamento","Balanceamento de carga (2.8)",b+"\n### Valores por rank\n"+dump)

if __name__=="__main__":
    fns={"serial":serial,"times":times,"strong":strong,"weak":weak,"comm":comm,"mpip":mpip,"perf":perf,"balance":balance}
    for a in sys.argv[1:]:
        try: fns[a]()
        except SystemExit as e: print(f"[{a}] {e}")
        except Exception as e: print(f"[{a}] erro: {e}")
PYEOF
  rm -f "$TAB"/*.md
  OUT="$OUT" P_LIST="$P_LIST" W="$W" H="$H" MPIP_P="$MPIP_P" F_SERIAL="$F_SERIAL" \
    python3 "$OUT/analyze.py" serial times strong weak comm mpip perf balance > "$OUT/analise_stdout.txt"
  cat "$TAB"/*.md > "$OUT/RELATORIO_TABELAS.md"
  log "Tabelas: $OUT/RELATORIO_TABELAS.md | CSV: $OUT/dados_strong_scaling.csv | gráfico: $OUT/speedup_strong.png"
fi
log "FIM"