/*
 * Game of Life com MPI - decomposicao de dominio por COLUNAS
 * Comunicacao NAO-BLOQUEANTE (MPI_Isend/MPI_Irecv + MPI_Waitall) com
 * sobreposicao de comunicacao e computacao (miolo vs bordas).
 *
 * OBJETIVO: manter o nucleo de computacao IDENTICO ao gol.c serial e ao
 * gol_mpi.c bloqueante, alterando apenas a estrategia de comunicacao para
 * permitir o calculo das colunas internas (2..lw-1) enquanto as colunas-fantasma
 * trafegam pela rede.
 *
 * Uso: gol_mpi_opt [largura altura iteracoes]   (padrao: 500 500 50000)
 * Compilar: mpicc -O2 -g -Wall gol_mpi_opt.c -o programa_otimizado
 * Executar: mpirun -np 4 ./programa_otimizado
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>   /* gethostname (Linux) */
#include <mpi.h>

#ifdef SEM_MODULO_X
#define XIDX(x1) (x1)                              /* otimizado (nao comparavel) */
#else
#define XIDX(x1) (((x1) + lw + 2) % (lw + 2))      /* paridade com (x1+w)%w */
#endif

/* Inicia a troca nao-bloqueante das colunas de borda com os vizinhos.
 * A coluna nao e contigua em row-major, entao usa-se o mesmo tipo derivado
 * (h blocos de 1 unsigned, passo lw+2). */
static void exchange_halo_start(void *u, int lw, int h, MPI_Datatype coltype,
                                int size, int left, int right,
                                MPI_Request reqs[4])
{
    unsigned (*univ)[lw + 2] = u;

    if (size == 1) {
        /* P = 1: nenhuma mensagem; o "vizinho" e o proprio dominio */
        for (int y = 0; y < h; y++) {
            univ[y][0]      = univ[y][lw];
            univ[y][lw + 1] = univ[y][1];
        }
        return;
    }
    /* Posta as recepcoes nas colunas-fantasma (direita = lw+1, esquerda = 0) */
    MPI_Irecv(&univ[0][lw + 1], 1, coltype, right, 0, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(&univ[0][0],      1, coltype, left,  1, MPI_COMM_WORLD, &reqs[1]);

    /* Posta os envios das colunas reais de borda (1a = 1, ultima = lw) */
    MPI_Isend(&univ[0][1],      1, coltype, left,  0, MPI_COMM_WORLD, &reqs[2]);
    MPI_Isend(&univ[0][lw],     1, coltype, right, 1, MPI_COMM_WORLD, &reqs[3]);
}

/* Fase 1 da evolucao (sobreposta ao Isend/Irecv): calcula apenas as colunas
 * internas (2 .. lw-1), que NAO dependem das colunas-fantasma (0 e lw+1). */
static void evolve_interior(void *u, void *nwp, int lw, int h)
{
    unsigned (*univ)[lw + 2] = u;
    unsigned (*nw)[lw + 2]   = nwp;

    if (lw <= 2) return;

    for (int y = 0; y < h; y++) {
        for (int x = 2; x <= lw - 1; x++) {
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

/* Fase 2 da evolucao (apos MPI_Waitall): calcula as colunas de borda (1 e lw),
 * que dependem das colunas-fantasma recem-recebidas, e copia o buffer "nw"
 * de volta para "univ" (garantindo que os buffers de envio nao sejam alterados
 * antes do termino do MPI_Waitall). */
static void evolve_borders_and_copy(void *u, void *nwp, int lw, int h)
{
    unsigned (*univ)[lw + 2] = u;
    unsigned (*nw)[lw + 2]   = nwp;
    int border_cols[2] = {1, lw};
    int n_borders = (lw > 1) ? 2 : 1;

    for (int y = 0; y < h; y++) {
        for (int i = 0; i < n_borders; i++) {
            int x = border_cols[i];
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
    int params[3] = {500, 500, 50000};              /* w, h, max_iter */
    if (rank == 0 && argc == 4) {
        params[0] = atoi(argv[1]);
        params[1] = atoi(argv[2]);
        params[2] = atoi(argv[3]);
    }
    MPI_Bcast(params, 3, MPI_INT, 0, MPI_COMM_WORLD);
    int w = params[0], h = params[1], max_iter = params[2];

    if (w < 1 || h < 1 || max_iter < 0 || size > w) {
        if (rank == 0)
            fprintf(stderr, "Parametros invalidos (P=%d, largura=%d). "
                            "E preciso P <= largura.\n", size, w);
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

    /* ---- Fase 2: laco principal (com sobreposicao comm/comp) ---- */
    MPI_Request reqs[4];
    for (int iter = 0; iter <= max_iter; iter++) {
        /* 1. Inicia a troca de colunas-fantasma (MPI_Irecv + MPI_Isend) */
        t0 = MPI_Wtime();
        exchange_halo_start(univ, lw, h, coltype, size, left, right, reqs);
        if (size > 1) {                      /* P > 1: inicio do Isend/Irecv -> TCOMM */
            t_comm += MPI_Wtime() - t0;
        } else {                             /* P = 1: copia local, sem MPI -> TCOMP */
            t_comp += MPI_Wtime() - t0;
        }

        /* 2. Calcula as colunas internas (2..lw-1) enquanto a rede transmite */
        t0 = MPI_Wtime();
        evolve_interior(univ, nw, lw, h);
        t_comp += MPI_Wtime() - t0;

        /* 3. Aguarda a conclusao das trocas de borda (MPI_Waitall) */
        if (size > 1) {
            t0 = MPI_Wtime();
            MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
            t_comm += MPI_Wtime() - t0;
            bytes_sent += 2LL * h * (long long)sizeof(unsigned);
        }

        /* 4. Calcula as colunas de borda (1 e lw) e copia nw -> univ */
        t0 = MPI_Wtime();
        evolve_borders_and_copy(univ, nw, lw, h);
        t_comp += MPI_Wtime() - t0;
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
    double mine[5]   = {t_init, t_comm, t_comp, t_reduce, t_total};
    double maxima[5] = {0};
    MPI_Reduce(mine, maxima, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Grade %dx%d | %d iteracoes | %d processo(s)\n", w, h, max_iter, size);
        printf("Total de celulas vivas ao final: %lld\n", total_alive);
        printf("--- Tempos (maximo entre ranks, em segundos) ---\n");
        printf("Inicializacao : %f\n", maxima[0]);
        printf("Comunicacao   : %f\n", maxima[1]);
        printf("Computacao    : %f\n", maxima[2]);
        printf("Reducao final : %f\n", maxima[3]);
        printf("Tempo total   : %f\n", maxima[4]);
    }

    MPI_Type_free(&coltype);
    free(univ);
    free(nw);
    MPI_Finalize();
    return 0;
}