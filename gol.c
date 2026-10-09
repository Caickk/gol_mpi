#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

/*
// Função para gerar o arquivo PBM
void save_pbm(void *u, int w, int h, int iter)
{
   int (*univ)[w] = u;
   char filename[64];
   
   // Formata o nome do arquivo para gol_nn.pbm (ex: gol_0, gol_500, gol_1000)
   sprintf(filename, "gol_%d.pbm", iter);
   
   FILE *f = fopen(filename, "w");
   if (!f) {
      perror("Erro ao criar o arquivo PBM");
      return;
   }

   // Cabeçalho e dados PBM no formato P1 (ASCII monocromático)
   fprintf(f, "P1\n%d %d\n", w, h);
   for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
         fprintf(f, "%d ", univ[y][x]);
      }
      fprintf(f, "\n");
   }
   fclose(f);
}
*/

// Função principal que calcula a próxima geração do Game of Life.
void evolve(void *u, int w, int h)
{
   unsigned (*univ)[w] = u;
   unsigned new[h][w];

   for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
         int n = 0;
         
         for (int y1 = y - 1; y1 <= y + 1; y1++) {
            for (int x1 = x - 1; x1 <= x + 1; x1++) {
               if (univ[(y1 + h) % h][(x1 + w) % w]) {
                  n++;
               }
            }
         }

         if (univ[y][x]) n--;
         
         new[y][x] = (n == 3 || (n == 2 && univ[y][x]));
      }
   }

   for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
         univ[y][x] = new[y][x];
      }
   }
}

int count_alive(void *u, int w, int h)
{
    unsigned (*univ)[w] = u;
    int total_alive = 0;
    // Leitura estritamente sequencial, linha por coluna (sem paralelismo)
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            total_alive += univ[y][x];
        }
    }
    return total_alive;
}

// A função game recebe largura, altura da matriz e o total de iterações 
void game(int w, int h, int max_iter)
{
   unsigned univ[h][w];
   
   // Inicializa o tabuleiro: Uma cruz perfeita cruzando a matriz ao meio
   for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
         // Ativa a célula se ela estiver na linha do meio OU na coluna do meio
         if (x == w / 2 || y == h / 2) {
            univ[y][x] = 1;
         } else {
            univ[y][x] = 0;
         }
      }
   }
   
   // const int save_interval = 500; // Salva a cada 500 iterações
   for (int iter = 0; iter <= max_iter; iter++) {
      // if (iter % save_interval == 0) {
      //    save_pbm(univ, w, h, iter);
      // }
      evolve(univ, w, h);
   }
   int final_alive = count_alive(univ, w, h);
   printf("Total de celulas vivas ao final: %d\n", final_alive);
}

int main(void)
{
   int w = 500;
   int h = 500;
   int max_iter = 50000;

   struct timespec start, end;

   // Captura o tempo EXATAMENTE ANTES do processamento iniciar
   clock_gettime(CLOCK_MONOTONIC, &start);

   game(w, h, max_iter);

   // Captura o tempo EXATAMENTE APÓS o processamento terminar
   clock_gettime(CLOCK_MONOTONIC, &end);

   double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;

   printf("Tempo interno de execucao: %f segundos\n", time_taken);
   
   return 0;
}