# Análise de Desempenho Paralelo com MPI

> **Objetivo:** documentar o algoritmo, a implementação serial e paralela, a metodologia experimental e a análise dos resultados, incluindo a comparação com OpenMP e a avaliação da otimização proposta.
>
> **Orientação:** substitua os campos entre colchetes pelos dados medidos e inclua as saídas reais das ferramentas. Não preencha lacunas com valores estimados.

## 1. Algoritmo, volume de dados e decomposição paralela

### 1.1 Descrição do algoritmo

[Descreva o problema resolvido, a entrada, a saída e as principais etapas do algoritmo.]

### 1.2 Volume de dados

| Parâmetro | Valor |
|---|---|
| Tamanho da entrada (`N`) | [preencher] |
| Tipo e tamanho dos elementos | [preencher] |
| Volume total de dados | [preencher] |
| Critério de validação do resultado | [preencher] |

Explique como o volume de dados foi definido e se permaneceu constante nos experimentos de *strong scaling* ou variou nos experimentos de *weak scaling*.

### 1.3 Decomposição paralela

[Explique como o trabalho é dividido entre os processos MPI, como os dados são distribuídos, quais operações de comunicação coletiva ou ponto a ponto são utilizadas e como os resultados parciais são reunidos.]

- **Estratégia de decomposição:** [por dados / por tarefas / outra]
- **Distribuição de dados:** [preencher]
- **Comunicação principal:** [preencher]
- **Tratamento de divisão desigual:** [preencher, se aplicável]

## 2. Código-fonte: versões serial, MPI e MPI otimizada

### 2.1 Versão serial

Arquivo: [`[caminho-do-arquivo-serial]`]( [caminho-do-arquivo-serial] )

```c
/* Inclua aqui o código serial ou forneça o caminho para o arquivo. */
```

### 2.2 Versão MPI original

Arquivo: [`[caminho-do-arquivo-mpi]`]( [caminho-do-arquivo-mpi] )

```c
/* Inclua aqui o código MPI original ou forneça o caminho para o arquivo. */
```

### 2.3 Versão MPI otimizada

Arquivo: [`[caminho-do-arquivo-mpi-otimizado]`]( [caminho-do-arquivo-mpi-otimizado] )

```c
/* Inclua aqui o código MPI otimizado ou forneça o caminho para o arquivo. */
```

### 2.4 Compilação e execução

```bash
# Exemplo; adapte compilador, opções, arquivos e parâmetros ao projeto.
mpicc -O3 -o programa_mpi programa_mpi.c
mpirun -np <P> ./programa_mpi <argumentos>
```

Registre a versão do compilador, as opções de compilação, a implementação MPI, os parâmetros de execução e o ambiente utilizado.

## 3. Tabelas de medição: `MPI_Wtime` e `/usr/bin/time -v`

### 3.1 Metodologia

Descreva como o tempo foi medido com `MPI_Wtime`, incluindo os pontos de início e fim, o uso de barreiras (se houver), o número de repetições e o tratamento da execução de aquecimento. Informe se os tempos correspondem apenas à região computacional ou ao tempo total da aplicação.

Para cada execução, registre o tempo por rank. Para resumir uma execução paralela, use:

- **Tmax:** maior tempo entre os ranks.
- **Tmin:** menor tempo entre os ranks.
- **Tmédia:** média aritmética dos tempos dos ranks.
- **Desvio padrão:** dispersão dos tempos entre os ranks, indicando se foi calculado como população ou amostra.

> O tempo de conclusão da execução paralela costuma ser representado por `Tmax`, pois a aplicação depende da conclusão do rank mais lento. Esclareça a definição usada no experimento.

### 3.2 `MPI_Wtime` por rank

| P | Repetição | Rank | Tempo (s) |
|---:|---:|---:|---:|
| [P] | [r] | [rank] | [tempo] |

#### Resumo por configuração

| P | Repetições | Tmax médio (s) | Tmin médio (s) | Tmédia dos ranks (s) | Desvio padrão (s) |
|---:|---:|---:|---:|---:|---:|
| [P] | [n] | [valor] | [valor] | [valor] | [valor] |

Explique como os resultados de diferentes repetições foram agregados. Diferencie o desvio entre ranks do desvio entre repetições, se ambos forem apresentados.

### 3.3 `/usr/bin/time -v`

| P | Repetição | Tempo real (s) | Tempo de usuário (s) | Tempo de sistema (s) | CPU (%) | RSS máximo (unidade) | Falhas de página | Trocas de contexto |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [P] | [r] | [valor] | [valor] | [valor] | [valor] | [valor] | [valor] | [valor] |

Comando de exemplo:

```bash
/usr/bin/time -v mpirun -np <P> ./programa_mpi <argumentos>
```

#### Limitações de `/usr/bin/time` com o launcher

Quando executado sobre `mpirun` ou outro launcher, `/usr/bin/time -v` mede o processo que foi iniciado diretamente — normalmente o launcher — e não necessariamente cada processo MPI individual. Assim, seus dados de CPU, memória residente e falhas de página podem não representar a soma nem o comportamento isolado dos ranks. A forma como os processos filhos são contabilizados depende da ferramenta, do launcher e do ambiente.

Para caracterizar cada rank, use medições dentro da aplicação ou execute a ferramenta de medição no processo MPI apropriado. Informe explicitamente qual processo foi medido e evite comparar métricas de escopos diferentes como se fossem equivalentes.

## 4. *Strong scaling*: desempenho e Lei de Amdahl

No *strong scaling*, o tamanho total do problema permanece constante enquanto o número de processos `P` aumenta.

### 4.1 Resultados

| P | Tmax médio (s) | Desvio de Tmax (s) | Speedup médio | Desvio do speedup | Eficiência média (%) | Desvio da eficiência (%) |
|---:|---:|---:|---:|---:|---:|---:|
| [P] | [valor] | [valor] | [valor] | [valor] | [valor] | [valor] |

Use uma definição consistente para o tempo serial de referência `T1`:

- `Speedup(P) = T1 / Tmax(P)`
- `Eficiência(P) = Speedup(P) / P × 100%`

Informe se `T1` é o tempo da versão serial ou o tempo MPI executado com um processo. Não misture referências diferentes na mesma tabela.

### 4.2 Gráfico de *strong scaling*

Insira um gráfico com `P` no eixo horizontal e speedup no eixo vertical. Inclua os resultados medidos, barras de erro quando apropriado e a curva ideal `Speedup(P) = P`.

[Inserir gráfico ou link para o arquivo do gráfico.]

### 4.3 Comparação com a Lei de Amdahl

A Lei de Amdahl estima o speedup máximo quando uma fração `s` do trabalho é serial:

\[
S(P) = \frac{1}{s + \frac{1-s}{P}}
\]

Estime ou justifique a fração serial `s` com base no algoritmo e nos resultados; não a escolha apenas para ajustar a curva. Compare a curva teórica com os valores medidos e discuta diferenças causadas por comunicação, sincronização, desequilíbrio de carga e custos de execução.

## 5. *Weak scaling*

No *weak scaling*, o volume de trabalho por processo deve permanecer aproximadamente constante, de modo que o tamanho total `Ntotal` cresce com `P`.

| P | Ntotal | Dados por processo | Tmax médio (s) | Desvio de Tmax (s) |
|---:|---:|---:|---:|---:|
| [P] | [valor] | [valor] | [valor] | [valor] |

Descreva como `Ntotal` foi calculado e como os dados foram distribuídos. Discuta se o tempo permanece aproximadamente constante conforme `P` aumenta e explique possíveis desvios por comunicação, sincronização, memória e infraestrutura.

## 6. Comunicação versus computação

Explique como `Tcomp` e `Tcomm` foram medidos e se os intervalos são mutuamente exclusivos. Especifique quais operações foram contabilizadas como comunicação e se os tempos incluem sincronização ou espera.

| P | Tcomp médio (s) | Tcomm médio (s) | Fcomm (%) |
|---:|---:|---:|---:|
| [P] | [valor] | [valor] | [valor] |

Definição sugerida, se compatível com a metodologia adotada:

\[
F_{comm} = \frac{T_{comm}}{T_{comp} + T_{comm}} \times 100\%
\]

Explique a tendência observada conforme `P` aumenta e relacione-a com o speedup e a eficiência. Se os tempos se sobrepõem ou foram obtidos por métodos distintos, descreva essa limitação antes de calcular a fração.

## 7. Relatório do mpiP

### 7.1 Saída real

Inclua a saída original do mpiP, preservando os cabeçalhos, as unidades e as estatísticas relevantes. Remova apenas informações sensíveis ou irrelevantes, se necessário.

```text
[COLE AQUI A SAÍDA REAL DO mpiP]
```

Registre a versão do mpiP, o comando de execução, o número de ranks e as condições do experimento.

### 7.2 Análise das três chamadas MPI mais caras

| Posição | Chamada MPI | Tempo acumulado (s) | % do tempo MPI | Número de chamadas | Interpretação |
|---:|---|---:|---:|---:|---|
| 1 | [chamada] | [valor] | [valor] | [valor] | [análise] |
| 2 | [chamada] | [valor] | [valor] | [valor] | [análise] |
| 3 | [chamada] | [valor] | [valor] | [valor] | [análise] |

Identifique as três chamadas de maior custo conforme uma métrica claramente indicada (por exemplo, tempo acumulado). Explique o papel de cada chamada, se o custo é esperado e quais alterações poderiam reduzi-lo. Não confunda tempo acumulado com tempo médio por chamada.

## 8. `perf stat` por nó: IPC e cache misses

### 8.1 Metodologia e saída real

Informe o comando utilizado, a versão do `perf`, o nó ou máquina avaliada, o processo monitorado e os eventos habilitados. Inclua a saída real de cada nó.

**Nó [identificação]:**

```text
[COLE AQUI A SAÍDA REAL DO perf stat]
```

Repita o bloco para cada nó ou ambiente avaliado.

### 8.2 Tabela comparativa

| Nó | P | Ciclos | Instruções | IPC | Cache references | Cache misses | Taxa de cache misses (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| [nó] | [P] | [valor] | [valor] | [valor] | [valor] | [valor] | [valor] |

Definições, quando os eventos estiverem disponíveis e forem comparáveis:

- `IPC = instruções / ciclos`
- `Taxa de cache misses = cache misses / cache references × 100%`

Verifique os nomes e a disponibilidade dos eventos na plataforma. Se a contagem for multiplexada, escalada ou indisponível, registre isso. Não compare diretamente eventos diferentes entre nós sem explicar as diferenças de hardware e configuração.

### 8.3 Análise

Discuta o IPC como indicador agregado de instruções concluídas por ciclo, sem tratá-lo isoladamente como medida absoluta de desempenho. Relacione cache misses, ciclos, instruções, tempo de execução e características do algoritmo. Diferencie métricas coletadas por nó de métricas coletadas por rank ou processo.

## 9. Balanceamento de carga por rank

### 9.1 Valores por rank

| P | Rank | Tempo de computação (s) | Tempo de comunicação (s) | Tempo total medido (s) | Trabalho atribuído |
|---:|---:|---:|---:|---:|---:|
| [P] | [rank] | [valor] | [valor] | [valor] | [quantidade] |

### 9.2 Análise

Compare os tempos e o trabalho atribuído aos ranks. Identifique ranks mais lentos, diferenças na distribuição de dados e períodos de espera. Relacione o rank mais lento a `Tmax` e discuta se o desequilíbrio é causado pela decomposição, por comunicação, por diferenças entre nós ou por variação experimental.

Explique a métrica de desequilíbrio adotada. Por exemplo, pode-se comparar o maior tempo de computação com a média dos ranks, desde que a fórmula e a interpretação sejam explicitadas.

## 10. Comparação OpenMP versus MPI

Integre aqui os dados do trabalho anterior, mantendo condições comparáveis de hardware, tamanho de entrada, número de threads/processos, compilação e critério de tempo.

| Modelo | Configuração | Tempo médio (s) | Desvio padrão (s) | Speedup | Eficiência (%) | Observações |
|---|---|---:|---:|---:|---:|---|
| Serial | 1 processo / 1 thread | [valor] | [valor] | 1,00 | 100,0 | [referência] |
| OpenMP | [n threads] | [valor] | [valor] | [valor] | [valor] | [trabalho anterior] |
| MPI | [P ranks] | [valor] | [valor] | [valor] | [valor] | [este trabalho] |

Declare a referência usada para speedup e eficiência. Caso OpenMP e MPI tenham sido avaliados em máquinas, ambientes ou configurações diferentes, deixe a limitação explícita e evite atribuir toda a diferença ao modelo de paralelismo.

Discuta as diferenças entre memória compartilhada (OpenMP) e memória distribuída (MPI), incluindo comunicação, sincronização, custo de criação de threads/processos e limites de escalabilidade.

## 11. Otimização proposta e comparação original versus otimizado

### 11.1 Descrição da otimização

[Descreva a alteração proposta, o problema que ela pretende resolver e por que se espera que melhore o desempenho.]

### 11.2 Justificativa baseada nos dados

Use evidências das seções anteriores: chamadas MPI mais caras, proporção de comunicação, desequilíbrio entre ranks, IPC, cache misses ou escalabilidade. Relacione cada evidência com a otimização escolhida e indique quais métricas devem mudar caso a hipótese esteja correta.

### 11.3 Comparação experimental

| P | Versão | Tmax médio (s) | Desvio de Tmax (s) | Speedup relativo | Comunicação (%) | Validação correta? |
|---:|---|---:|---:|---:|---:|---|
| [P] | Original | [valor] | [valor] | 1,00 | [valor] | [sim/não] |
| [P] | Otimizada | [valor] | [valor] | [valor] | [valor] | [sim/não] |

Use as mesmas entradas, condições de execução e critérios de medição nas duas versões. Defina `Speedup relativo = Tmax_original / Tmax_otimizado`, se esse for o critério escolhido. Inclua mais de uma configuração de `P` quando possível.

### 11.4 Discussão

Informe o ganho ou a perda observada, sua variação entre repetições e possíveis custos introduzidos pela otimização. Confirme que a versão otimizada produz resultados corretos. Se a melhoria não for estatisticamente clara, registre isso em vez de afirmar um ganho conclusivo.

## 12. Conclusões

Resuma os principais resultados e responda:

1. O que foi aprendido sobre a implementação em memória distribuída?
2. Como `P` afetou o tempo, o speedup e a eficiência?
3. Em que condições o custo de comunicação passou a limitar o desempenho?
4. O que as medições por rank revelaram sobre o balanceamento de carga?
5. Como os resultados de MPI se comparam aos de OpenMP, considerando as limitações experimentais?
6. A otimização proposta trouxe benefício mensurável e manteve a correção?

Finalize com as limitações do experimento e possíveis melhorias futuras, como testar outros tamanhos de entrada, mais configurações de processos, outras estratégias de decomposição ou medições mais detalhadas por rank.

---

## Reprodutibilidade

Registre as informações necessárias para repetir os experimentos:

- **Sistema operacional e kernel:** [preencher]
- **Nós e hardware:** [preencher]
- **Implementação e versão MPI:** [preencher]
- **Compilador e opções:** [preencher]
- **Versões de mpiP e perf:** [preencher]
- **Comandos completos:** [preencher]
- **Tamanho da entrada e parâmetros:** [preencher]
- **Número de repetições e critério de agregação:** [preencher]
- **Localização dos códigos, logs e gráficos:** [preencher]
