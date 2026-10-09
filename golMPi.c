/*
 * Game of Life com MPI - decomposicao de dominio por COLUNAS
 * Comunicacao bloqueante (MPI_Sendrecv) para troca de colunas-fantasma.
 *
 * Fronteira toroidal (igual ao codigo sequencial): as bordas "dao a volta".
 *  - Vertical: tratada localmente (cada rank tem todas as linhas).
 *  - Horizontal: rank 0 e vizinho a esquerda do rank P-1 (anel de processos).
 *
 * Layout da memoria: COLUNA-MAIOR. A coluna c ocupa H posicoes contiguas,
 * logo cada coluna pode ser enviada diretamente, sem empacotar e sem
 * tipos derivados.  Cada rank guarda lw colunas reais + 2 colunas-fantasma:
 *    indice 0      -> fantasma esquerda
 *    indices 1..lw -> colunas reais do rank
 *    indice lw+1   -> fantasma direita
 *
 * Uso: gol_mpi [largura altura iteracoes]   (padrao: 500 500 50000)
 * Windows (MSVC + MS-MPI):
 *   cl /O2 gol_mpi.c /I"%MSMPI_INC%" /link /LIBPATH:"%MSMPI_LIB64%" msmpi.lib
 *   mpiexec -n 4 gol_mpi.exe
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define CELL(g, c, y, H) ((g)[(size_t)(c) * (H) + (y)])

/* Troca as colunas de borda com os vizinhos (bloqueante). */
static void exchange_halo(unsigned char *g, int H, int lw,
                          int size, int left, int right)
{
    if (size == 1) {
        /* P = 1: nenhuma mensagem; o "vizinho" e o proprio dominio */
        memcpy(&g[0],                    &g[(size_t)lw * H], H);
        memcpy(&g[(size_t)(lw + 1) * H], &g[(size_t)1 * H],  H);
        return;
    }
    /* 1) envio minha 1a coluna real para a esquerda,
          recebo a 1a coluna do vizinho da direita -> fantasma direita */
    MPI_Sendrecv(&g[(size_t)1 * H],        H, MPI_UNSIGNED_CHAR, left,  0,
                 &g[(size_t)(lw + 1) * H], H, MPI_UNSIGNED_CHAR, right, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    /* 2) envio minha ultima coluna real para a direita,
          recebo a ultima do vizinho da esquerda -> fantasma esquerda */
    MPI_Sendrecv(&g[(size_t)lw * H], H, MPI_UNSIGNED_CHAR, right, 1,
                 &g[0],              H, MPI_UNSIGNED_CHAR, left,  1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

/* Calcula a proxima geracao das colunas reais: cur -> nxt */
static void evolve(const unsigned char *cur, unsigned char *nxt,
                   int H, int lw, const int *yu, const int *yd)
{
    for (int c = 1; c <= lw; c++) {
        for (int y = 0; y < H; y++) {
            int a = yu[y], b = yd[y];
            int n = CELL(cur, c - 1, a, H) + CELL(cur, c - 1, y, H) + CELL(cur, c - 1, b, H)
                  + CELL(cur, c,     a, H)                          + CELL(cur, c,     b, H)
                  + CELL(cur, c + 1, a, H) + CELL(cur, c + 1, y, H) + CELL(cur, c + 1, b, H);
            CELL(nxt, c, y, H) = (unsigned char)
                (n == 3 || (n == 2 && CELL(cur, c, y, H)));
        }
    }
}

int main(int argc, char **argv)
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ---- Parametros: rank 0 define, MPI_Bcast distribui ---- */
    int params[3] = {500, 500, 50000};              /* W, H, max_iter */
    if (rank == 0 && argc == 4) {
        params[0] = atoi(argv[1]);
        params[1] = atoi(argv[2]);
        params[2] = atoi(argv[3]);
    }
    MPI_Bcast(params, 3, MPI_INT, 0, MPI_COMM_WORLD);
    int W = params[0], H = params[1], max_iter = params[2];

    if (W < 1 || H < 1 || max_iter < 0 || size > W) {
        if (rank == 0)
            fprintf(stderr, "Parametros invalidos (P=%d, largura=%d). "
                            "E preciso P <= largura.\n", size, W);
        MPI_Finalize();
        return 1;
    }

    /* ---- Decomposicao deterministica por colunas ---- */
    int base  = W / size, rem = W % size;
    int lw    = base + (rank < rem ? 1 : 0);        /* colunas deste rank   */
    int start = rank * base + (rank < rem ? rank : rem); /* 1a coluna global */
    int left  = (rank - 1 + size) % size;
    int right = (rank + 1) % size;

    double t_init = 0.0, t_comm = 0.0, t_comp = 0.0, t_reduce = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    /* ---- Fase 1: alocacao e inicializacao ---- */
    double t0 = MPI_Wtime();

    size_t bytes = (size_t)(lw + 2) * H;
    unsigned char *cur = calloc(bytes, 1);
    unsigned char *nxt = calloc(bytes, 1);
    int *yu = malloc(H * sizeof(int));              /* linha acima (toroidal)  */
    int *yd = malloc(H * sizeof(int));              /* linha abaixo (toroidal) */
    if (!cur || !nxt || !yu || !yd) {
        fprintf(stderr, "Rank %d: falha de alocacao\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int y = 0; y < H; y++) {
        yu[y] = (y - 1 + H) % H;
        yd[y] = (y + 1) % H;
    }
    /* cruz: coluna do meio OU linha do meio (coluna global = start + c - 1) */
    for (int c = 1; c <= lw; c++) {
        int gx = start + c - 1;
        for (int y = 0; y < H; y++)
            CELL(cur, c, y, H) = (gx == W / 2 || y == H / 2);
    }
    t_init = MPI_Wtime() - t0;

    /* ---- Fase 2: laco principal (mesmo numero de geracoes do original) ---- */
    for (int iter = 0; iter <= max_iter; iter++) {
        t0 = MPI_Wtime();
        exchange_halo(cur, H, lw, size, left, right);
        t_comm += MPI_Wtime() - t0;

        t0 = MPI_Wtime();
        evolve(cur, nxt, H, lw, yu, yd);
        t_comp += MPI_Wtime() - t0;

        unsigned char *tmp = cur; cur = nxt; nxt = tmp;   /* double buffering */
    }

    /* ---- Fase 3: contagem global de celulas vivas (MPI_Reduce) ---- */
    t0 = MPI_Wtime();
    long long local_alive = 0, total_alive = 0;
    for (int c = 1; c <= lw; c++)
        for (int y = 0; y < H; y++)
            local_alive += CELL(cur, c, y, H);
    MPI_Reduce(&local_alive, &total_alive, 1, MPI_LONG_LONG, MPI_SUM,
               0, MPI_COMM_WORLD);
    t_reduce = MPI_Wtime() - t0;

    double t_total = MPI_Wtime() - t_start;

    /* ---- Tempos: maximo entre ranks de cada fase (MPI_Reduce) ---- */
    double mine[5]   = {t_init, t_comm, t_comp, t_reduce, t_total};
    double maxima[5] = {0};
    MPI_Reduce(mine, maxima, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Grade %dx%d | %d iteracoes | %d processo(s)\n",
               W, H, max_iter, size);
        printf("Total de celulas vivas ao final: %lld\n", total_alive);
        printf("--- Tempos (maximo entre ranks, em segundos) ---\n");
        printf("Inicializacao : %f\n", maxima[0]);
        printf("Comunicacao   : %f\n", maxima[1]);
        printf("Computacao    : %f\n", maxima[2]);
        printf("Reducao final : %f\n", maxima[3]);
        printf("Tempo total   : %f\n", maxima[4]);
    }

    free(cur); free(nxt); free(yu); free(yd);
    MPI_Finalize();
    return 0;
}