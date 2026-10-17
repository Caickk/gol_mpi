/*
 * Game of Life com MPI - decomposicao de dominio por COLUNAS
 * VERSAO OTIMIZADA: SOBREPOSICAO DE COMUNICACAO E COMPUTACAO
 * (derivada de golmpi.c, que continua sendo a versao original com MPI_Sendrecv)
 *
 * MOTIVACAO (profiling): o T_comm por rank e quase constante com P, pois cada
 * geracao paga a latencia de 2 MPI_Sendrecv de ~2 KB, enquanto o T_comp cai com
 * 1/P; assim a fracao de comunicacao cresce com P. Por geracao o calculo e
 * varias vezes maior que a troca, entao ha o que esconder.
 *
 * COMO FUNCIONA (P > 1), em cada geracao:
 *   1) MPI_Irecv das 2 colunas-fantasma e MPI_Isend das 2 colunas de borda;
 *   2) calcula o MIOLO (colunas 2..lw-1), que nao le as fantasmas;
 *   3) MPI_Waitall;
 *   4) calcula as colunas de borda 1 e lw (agora com as fantasmas);
 *   5) copia nw -> univ (so depois do Waitall: os envios leem univ).
 * A conta por celula e identica a do serial; so mudam os limites dos lacos.
 * O volume de dados trocado NAO muda (2 colunas por geracao por rank).
 * Com P = 1 nao ha mensagens: copia local das bordas e calculo completo.
 *
 * MEDICAO: TCOMM = tempo dentro de chamadas MPI = postagem (Irecv/Isend) +
 * Waitall + Reduce + Barrier. O tempo no Waitall e a comunicacao NAO escondida
 * pela computacao ("Espera Waitall" no resumo).
 *
 * OBJETIVO: manter o nucleo de computacao IDENTICO ao gol.c serial, para que
 * a comparacao de desempenho (profiling / speedup) seja justa. As unicas
 * diferencas em relacao ao serial sao as inerentes a paralelizacao:
 *   - cada rank guarda apenas lw colunas + 2 colunas-fantasma;
 *   - o vizinho horizontal vem da coluna-fantasma (no serial vinha de
 *     (x1 + w) % w); o vizinho vertical continua usando (y1 + h) % h;
 *   - ha troca de colunas entre ranks e medicao de tempo por fase.
 *
 * PARIDADE DO MODULO HORIZONTAL (XIDX): com colunas-fantasma o indice x1 ja
 * esta sempre dentro de [0, lw+1], entao o "% w" do serial deixaria de ser
 * necessario. Medido neste codigo, remove-lo torna o calculo ~3x mais rapido
 * (a divisao inteira domina o custo por celula), o que inflaria o speedup.
 * Por padrao, XIDX mantem uma operacao de modulo equivalente (valor inalterado,
 * custo equivalente ao serial). Compile com -DSEM_MODULO_X para desativar.
 *
 * Mesmos elementos do serial: tipo unsigned (4 bytes), matriz row-major,
 * acesso univ[y][x], 3x3 com subtracao da propria celula, modulo no indice
 * vertical, calculo em buffer "new" + copia de volta a cada geracao.
 *
 * Fronteira toroidal: o rank 0 e vizinho a esquerda do rank P-1 (anel).
 * Layout local: univ[h][lw + 2]
 *    coluna 0      -> fantasma esquerda
 *    colunas 1..lw -> colunas reais do rank
 *    coluna lw+1   -> fantasma direita
 *
 * Uso: golmpi [largura [altura [iteracoes]]]   (padrao: 500 500 50000)
 *      Parametros opcionais e posicionais; os omitidos mantem o valor padrao.
 * Compilar: mpicc -O2 -Wall gol_mpi.c -o gol_mpi
 * Executar: mpirun -np 4 ./gol_mpi
 */
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>   /* INT_MAX */
#include <unistd.h>   /* gethostname (Linux) */
#include <mpi.h>

#ifdef SEM_MODULO_X
#define XIDX(x1) (x1)                              /* otimizado (nao comparavel) */
#else
#define XIDX(x1) (((x1) + lw + 2) % (lw + 2))      /* paridade com (x1+w)%w */
#endif

/* P = 1: nenhuma mensagem; o "vizinho" e o proprio dominio (nao e chamada MPI) */
static void halo_local(void *u, int lw, int h)
{
    unsigned (*univ)[lw + 2] = u;
    for (int y = 0; y < h; y++) {
        univ[y][0]      = univ[y][lw];
        univ[y][lw + 1] = univ[y][1];
    }
}

/* P > 1: dispara a troca de colunas SEM bloquear. A coluna nao e contigua em
 * row-major, entao usa-se um tipo derivado (h blocos de 1 unsigned, passo lw+2).
 * Tags: 0 = "1a coluna real, indo para a esquerda"; 1 = "ultima, indo para a direita".
 * Distinguir por tag mantem o casamento correto com P = 2 (vizinho esq. = dir.). */
static void halo_start(void *u, int lw, MPI_Datatype coltype,
                       int left, int right, MPI_Request req[4])
{
    unsigned (*univ)[lw + 2] = u;

    MPI_Irecv(&univ[0][lw + 1], 1, coltype, right, 0, MPI_COMM_WORLD, &req[0]); /* fantasma direita  */
    MPI_Irecv(&univ[0][0],      1, coltype, left,  1, MPI_COMM_WORLD, &req[1]); /* fantasma esquerda */
    MPI_Isend(&univ[0][1],      1, coltype, left,  0, MPI_COMM_WORLD, &req[2]); /* 1a coluna real    */
    MPI_Isend(&univ[0][lw],     1, coltype, right, 1, MPI_COMM_WORLD, &req[3]); /* ultima coluna real*/
}

/* Calcula em "nw" a proxima geracao das colunas xa..xb (mesma logica do serial).
 * "nw" e um buffer auxiliar (equivalente ao "unsigned new[h][w]" do serial). */
static void compute_cols(void *u, void *nwp, int lw, int h, int xa, int xb)
{
    unsigned (*univ)[lw + 2] = u;
    unsigned (*nw)[lw + 2]   = nwp;

    if (xa > xb) return;
    for (int y = 0; y < h; y++) {
        for (int x = xa; x <= xb; x++) {
            int n = 0;
            for (int y1 = y - 1; y1 <= y + 1; y1++) {
                for (int x1 = x - 1; x1 <= x + 1; x1++) {
                    if (univ[(y1 + h) % h][XIDX(x1)]) {
                        n++;
                    }
                }
            }
            if (univ[y][x]) n--;
            nw[y][x] = (n == 3 || (n == 2 && univ[y][x]));
        }
    }
}

/* Copia de volta nw -> univ (colunas reais), como no serial */
static void copy_back(void *u, void *nwp, int lw, int h)
{
    unsigned (*univ)[lw + 2] = u;
    unsigned (*nw)[lw + 2]   = nwp;

    for (int y = 0; y < h; y++) {
        for (int x = 1; x <= lw; x++) {
            univ[y][x] = nw[y][x];
        }
    }
}

/* Soma local das celulas vivas (apenas colunas reais) */
static long long count_alive(void *u, int lw, int h)
{
    unsigned (*univ)[lw + 2] = u;
    long long total_alive = 0;
    for (int y = 0; y < h; y++) {
        for (int x = 1; x <= lw; x++) {
            total_alive += univ[y][x];
        }
    }
    return total_alive;
}

int main(int argc, char **argv)
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ---- Parametros: rank 0 define, MPI_Bcast distribui ---- */
    /* Opcionais e posicionais: [largura [altura [iteracoes]]].
     * Argumento omitido = valor padrao; sem argumentos roda 500 x 500, 50000. */
    int params[3] = {500, 500, 50000};              /* w, h, max_iter */
    if (rank == 0) {
        int bad = 0;
        if (argc > 4) {
            fprintf(stderr, "Argumentos demais (%d); o maximo e 3.\n", argc - 1);
            bad = 1;
        }
        for (int i = 1; i < argc && i <= 3; i++) {
            char *end;
            long v = strtol(argv[i], &end, 10);
            if (*argv[i] == '\0' || *end != '\0' || v < 0 || v > INT_MAX) {
                fprintf(stderr, "Argumento invalido: '%s' (esperado inteiro nao negativo).\n",
                        argv[i]);
                bad = 1;
            } else {
                params[i - 1] = (int)v;
            }
        }
        if (bad) params[0] = -1;     /* forca a recusa abaixo, em todos os ranks */
    }
    MPI_Bcast(params, 3, MPI_INT, 0, MPI_COMM_WORLD);
    int w = params[0], h = params[1], max_iter = params[2];

    if (w < 1 || h < 1 || max_iter < 0 || size > w) {
        if (rank == 0)
            fprintf(stderr, "Parametros invalidos (P=%d, largura=%d, altura=%d, iteracoes=%d).\n"
                            "Uso: %s [largura [altura [iteracoes]]]   (padrao: 500 500 50000)\n"
                            "Largura e altura >= 1, iteracoes >= 0 e P <= largura.\n",
                    size, w, h, max_iter, argv[0]);
        MPI_Finalize();
        return 1;
    }

    /* ---- Decomposicao deterministica por colunas ---- */
    int base  = w / size, rem = w % size;
    int lw    = base + (rank < rem ? 1 : 0);              /* colunas do rank   */
    int start = rank * base + (rank < rem ? rank : rem);  /* 1a coluna global  */
    int left  = (rank - 1 + size) % size;
    int right = (rank + 1) % size;

    /* ---- Identificacao do no: MPI_Gather para o rank 0 imprimir em ordem ---- */
    char hostname[256] = {0};
    gethostname(hostname, sizeof(hostname) - 1);
    char *all_hosts = NULL;
    int  *all_info  = NULL;
    int   my_info[2] = {start, lw};
    if (rank == 0) {
        all_hosts = malloc((size_t)size * 256);
        all_info  = malloc((size_t)size * 2 * sizeof(int));
    }
    MPI_Gather(hostname, 256, MPI_CHAR, all_hosts, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gather(my_info,  2,   MPI_INT,  all_info,  2,   MPI_INT,  0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (int r = 0; r < size; r++)
            printf("Rank %d/%d rodando em %s (colunas %d a %d)\n", r, size,
                   &all_hosts[(size_t)r * 256], all_info[2 * r],
                   all_info[2 * r] + all_info[2 * r + 1] - 1);
        fflush(stdout);
        free(all_hosts);
        free(all_info);
    }

    double t_init = 0.0, t_comm = 0.0, t_comp = 0.0, t_reduce = 0.0;
    double t_count = 0.0, t_barrier = 0.0;   /* contagem local (comp) e barreira final (comm) */
    double t_wait = 0.0;                     /* tempo no MPI_Waitall (comm nao escondida) */
    long long bytes_sent = 0;                /* carga util enviada por este rank */

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    /* ---- Fase 1: alocacao e inicializacao ---- */
    double t0 = MPI_Wtime();

    unsigned (*univ)[lw + 2] = calloc((size_t)h * (lw + 2), sizeof(unsigned));
    unsigned (*nw)[lw + 2]   = calloc((size_t)h * (lw + 2), sizeof(unsigned));
    if (!univ || !nw) {
        fprintf(stderr, "Rank %d: falha de alocacao\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Datatype coltype;                         /* uma coluna da matriz local */
    MPI_Type_vector(h, 1, lw + 2, MPI_UNSIGNED, &coltype);
    MPI_Type_commit(&coltype);

    /* cruz: coluna do meio OU linha do meio (coluna global = start + x - 1) */
    for (int y = 0; y < h; y++) {
        for (int x = 1; x <= lw; x++) {
            int gx = start + x - 1;
            if (gx == w / 2 || y == h / 2) {
                univ[y][x] = 1;
            } else {
                univ[y][x] = 0;
            }
        }
    }
    t_init = MPI_Wtime() - t0;

    /* ---- Fase 2: laco principal (mesmo numero de geracoes do serial) ---- */
    for (int iter = 0; iter <= max_iter; iter++) {
        if (size > 1) {
            MPI_Request req[4];
            double ta = MPI_Wtime();
            halo_start(univ, lw, coltype, left, right, req);   /* MPI: postagem      */
            double tb = MPI_Wtime();
            compute_cols(univ, nw, lw, h, 2, lw - 1);          /* miolo (sem fantasmas) */
            double tc = MPI_Wtime();
            MPI_Waitall(4, req, MPI_STATUSES_IGNORE);          /* MPI: espera residual  */
            double td = MPI_Wtime();
            compute_cols(univ, nw, lw, h, 1, 1);               /* borda esquerda     */
            if (lw > 1) compute_cols(univ, nw, lw, h, lw, lw); /* borda direita      */
            copy_back(univ, nw, lw, h);                        /* so apos o Waitall  */
            double te = MPI_Wtime();

            t_comm += (tb - ta) + (td - tc);                   /* chamadas MPI       */
            t_wait += td - tc;                                 /* parte exposta      */
            t_comp += (tc - tb) + (te - td);                   /* fora de MPI        */
            bytes_sent += 2LL * h * (long long)sizeof(unsigned);
        } else {                                               /* P = 1: sem MPI     */
            double ta = MPI_Wtime();
            halo_local(univ, lw, h);
            compute_cols(univ, nw, lw, h, 1, lw);
            copy_back(univ, nw, lw, h);
            t_comp += MPI_Wtime() - ta;
        }
    }

    /* ---- Fase 3: contagem global de celulas vivas (MPI_Reduce) ---- */
    t0 = MPI_Wtime();
    long long local_alive = count_alive(univ, lw, h);
    t_count = MPI_Wtime() - t0;                    /* calculo */

    long long total_alive = 0;
    t0 = MPI_Wtime();
    MPI_Reduce(&local_alive, &total_alive, 1, MPI_LONG_LONG, MPI_SUM,
               0, MPI_COMM_WORLD);
    t_reduce = MPI_Wtime() - t0;                   /* comunicacao */
    if (size > 1 && rank != 0) bytes_sent += (long long)sizeof(long long);

    t0 = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);                   /* barreira final */
    double t_end = MPI_Wtime();
    t_barrier = t_end - t0;                        /* comunicacao (espera) */

    double t_total = t_end - t_start;              /* TTOTAL: entre as barreiras */
    double tcomp = t_init + t_comp + t_count;      /* TCOMP: fora de chamadas MPI */
    double tcomm = t_comm + t_reduce + t_barrier;  /* TCOMM: dentro de chamadas MPI */

    /* Todos os ranks imprimem a propria linha (ordem de saida nao e garantida;
     * cada linha traz o RANK, entao basta ordenar depois). */
    printf("RANK %d HOST %s TCOMP %f TCOMM %f TTOTAL %f BYTES %lld\n",
           rank, hostname, tcomp, tcomm, t_total, bytes_sent);
    fflush(stdout);

    /* ---- Tempos: maximo entre ranks de cada fase (MPI_Reduce) ---- */
    double mine[6]   = {t_init, t_comm, t_comp, t_reduce, t_total, t_wait};
    double maxima[6] = {0};
    MPI_Reduce(mine, maxima, 6, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Grade %dx%d | %d iteracoes | %d processo(s)\n", w, h, max_iter, size);
        printf("Total de celulas vivas ao final: %lld\n", total_alive);
        printf("--- Tempos (maximo entre ranks, em segundos) ---\n");
        printf("Inicializacao : %f\n", maxima[0]);
        printf("Comunicacao   : %f\n", maxima[1]);
        printf("Computacao    : %f\n", maxima[2]);
        printf("Reducao final : %f\n", maxima[3]);
        printf("Tempo total   : %f\n", maxima[4]);
        printf("Espera Waitall: %f\n", maxima[5]);
    }

    MPI_Type_free(&coltype);
    free(univ);
    free(nw);
    MPI_Finalize();
    return 0;
}