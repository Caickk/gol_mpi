/*
 * Game of Life com MPI - decomposicao de dominio por COLUNAS
 * Comunicacao bloqueante (MPI_Sendrecv) para troca de colunas-fantasma.
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
 * Uso: gol_mpi [largura altura iteracoes]   (padrao: 500 500 50000)
 * Compilar: mpicc -O2 -Wall gol_mpi.c -o gol_mpi
 * Executar: mpirun -np 4 ./gol_mpi
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

/* Troca as colunas de borda com os vizinhos (bloqueante).
 * A coluna nao e contigua em row-major, entao usa-se um tipo derivado
 * (h blocos de 1 unsigned, passo lw+2) em vez de empacotar manualmente. */
static void exchange_halo(void *u, int lw, int h, MPI_Datatype coltype,
                          int size, int left, int right)
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
    /* 1) minha 1a coluna real vai para a esquerda;
          recebo a 1a coluna do vizinho da direita -> fantasma direita */
    MPI_Sendrecv(&univ[0][1],      1, coltype, left,  0,
                 &univ[0][lw + 1], 1, coltype, right, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    /* 2) minha ultima coluna real vai para a direita;
          recebo a ultima do vizinho da esquerda -> fantasma esquerda */
    MPI_Sendrecv(&univ[0][lw], 1, coltype, right, 1,
                 &univ[0][0],  1, coltype, left,  1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

/* Proxima geracao: mesma logica do serial. "nw" e um buffer auxiliar
 * (equivalente ao "unsigned new[h][w]" do serial), alocado uma unica vez. */
static void evolve(void *u, void *nwp, int lw, int h)
{
    unsigned (*univ)[lw + 2] = u;
    unsigned (*nw)[lw + 2]   = nwp;

    for (int y = 0; y < h; y++) {
        for (int x = 1; x <= lw; x++) {
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
        t0 = MPI_Wtime();
        exchange_halo(univ, lw, h, coltype, size, left, right);
        t_comm += MPI_Wtime() - t0;

        t0 = MPI_Wtime();
        evolve(univ, nw, lw, h);
        t_comp += MPI_Wtime() - t0;
    }

    /* ---- Fase 3: contagem global de celulas vivas (MPI_Reduce) ---- */
    t0 = MPI_Wtime();
    long long local_alive = count_alive(univ, lw, h);
    long long total_alive = 0;
    MPI_Reduce(&local_alive, &total_alive, 1, MPI_LONG_LONG, MPI_SUM,
               0, MPI_COMM_WORLD);
    t_reduce = MPI_Wtime() - t0;

    double t_total = MPI_Wtime() - t_start;

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