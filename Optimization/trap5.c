#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* ================= PARÂMETROS ================= */

#define N 20              // múltiplo de 5
#define BLOCK 5
#define POP_MAX 400
#define INIT_POP 100
#define GENERATIONS 500

#define MAX_PARENTS 2
#define SIM_THRESHOLD 2.5  // distância mínima aumentada
#define RANDOM_RATE 0.05
#define ELITE_SIZE 5

/* ================= ESTRUTURAS ================= */

typedef struct {
    int x[N];
    int fitness;
} Individual;

typedef struct {
    int parents[MAX_PARENTS];
    int num_parents;
} Node;

/* ================= GLOBAIS ================= */

Node bn[N];
double cpt[N][4][2]; // até 2 pais binário (máx 4 configurações)

/* ================= FITNESS ================= */

int trap5(int *x) {
    int total = 0;

    for (int i = 0; i < N; i += BLOCK) {
        int u = 0;
        for (int j = 0; j < BLOCK; j++)
            u += x[i + j];

        if (u == BLOCK)
            total += BLOCK;
        else
            total += (BLOCK - 1 - u);
    }

    return total;
}

int fitness(Individual *ind) {
    return trap5(ind->x);
}

/* ================= UTIL ================= */

void random_individual(Individual *ind) {
    for (int i = 0; i < N; i++)
        ind->x[i] = rand() % 2;
    ind->fitness = fitness(ind);
}

int cmp(const void *a, const void *b) {
    return ((Individual*)b)->fitness - ((Individual*)a)->fitness;
}

/* ================= DISTÂNCIA ================= */

double distance(Individual *a, Individual *b) {
    double d = 0;
    for (int i = 0; i < N; i++) {
        double diff = a->x[i] - b->x[i];
        d += diff * diff;
    }
    return sqrt(d);
}

/* ================= DETECÇÃO DE CICLOS ================= */

int has_cycle(int node, int parent, int visited[]) {
    if (visited[parent]) return 0;
    if (node == parent) return 1;
    
    visited[parent] = 1;
    for (int i = 0; i < bn[parent].num_parents; i++) {
        if (has_cycle(node, bn[parent].parents[i], visited))
            return 1;
    }
    return 0;
}

/* ================= BIC ================= */

double bic_score(Individual pop[], int size, int node, int parents[], int np) {
    int configs = 1 << np;
    int counts[4][2];
    
    // Inicializa com Laplace smoothing (pseudo-contagens)
    for (int i = 0; i < configs; i++)
        for (int j = 0; j < 2; j++)
            counts[i][j] = 1;

    for (int i = 0; i < size; i++) {
        int idx = 0;
        for (int p = 0; p < np; p++)
            idx = (idx << 1) | pop[i].x[parents[p]];
        if (idx < configs) {  // Segurança
            counts[idx][pop[i].x[node]]++;
        }
    }

    double loglik = 0;
    for (int i = 0; i < configs; i++) {
        int sum = counts[i][0] + counts[i][1];
        if (sum > 0) {
            for (int j = 0; j < 2; j++) {
                if (counts[i][j] > 0) {
                    loglik += counts[i][j] * log((double)counts[i][j] / sum);
                }
            }
        }
    }

    int k = configs;  // número de parâmetros
    return loglik - 0.5 * k * log(size);
}

/* ================= ESTRUTURA ================= */

void learn_structure(Individual pop[], int size) {
    for (int i = 0; i < N; i++) {
        bn[i].num_parents = 0;
        
        // Se a população for muito pequena, não aprendemos estrutura
        if (size < 10) continue;
        
        double best = bic_score(pop, size, i, NULL, 0);

        while (bn[i].num_parents < MAX_PARENTS) {
            int best_p = -1;

            for (int p = 0; p < N; p++) {
                if (p == i) continue;
                
                int used = 0;
                for (int k = 0; k < bn[i].num_parents; k++)
                    if (bn[i].parents[k] == p) used = 1;
                if (used) continue;

                // Verifica ciclo
                int visited[N] = {0};
                bn[i].parents[bn[i].num_parents] = p;
                bn[i].num_parents++;
                
                int creates_cycle = 0;
                // Verifica se algum nó tem ciclo
                for (int child = 0; child < N; child++) {
                    int vis[N] = {0};
                    if (has_cycle(child, child, vis)) {
                        creates_cycle = 1;
                        break;
                    }
                }
                
                bn[i].num_parents--;
                
                if (!creates_cycle) {
                    int temp[MAX_PARENTS];
                    for (int k = 0; k < bn[i].num_parents; k++)
                        temp[k] = bn[i].parents[k];
                    temp[bn[i].num_parents] = p;
                    
                    double score = bic_score(pop, size, i, temp, bn[i].num_parents + 1);
                    
                    if (score > best) {
                        best = score;
                        best_p = p;
                    }
                }
            }

            if (best_p >= 0) {
                bn[i].parents[bn[i].num_parents++] = best_p;
            } else {
                break;
            }
        }
    }
}

/* ================= PARÂMETROS ================= */

void learn_parameters(Individual pop[], int size) {
    for (int i = 0; i < N; i++) {
        int cfg = 1 << bn[i].num_parents;
        
        // Inicializa com Laplace smoothing (pseudo-contagens)
        for (int c = 0; c < cfg; c++) {
            cpt[i][c][0] = 1.0;
            cpt[i][c][1] = 1.0;
        }

        // Conta ocorrências
        for (int k = 0; k < size; k++) {
            int idx = 0;
            for (int p = 0; p < bn[i].num_parents; p++) {
                int parent_idx = bn[i].parents[p];
                idx = (idx << 1) | pop[k].x[parent_idx];
            }
            if (idx < cfg) {  // Segurança
                cpt[i][idx][pop[k].x[i]] += 1.0;
            }
        }

        // Normaliza para obter probabilidades
        for (int c = 0; c < cfg; c++) {
            double sum = cpt[i][c][0] + cpt[i][c][1];
            if (sum > 0) {
                cpt[i][c][0] /= sum;
                cpt[i][c][1] /= sum;
            } else {
                // Fallback para distribuição uniforme
                cpt[i][c][0] = 0.5;
                cpt[i][c][1] = 0.5;
            }
        }
        
        // Preenche configurações não utilizadas com distribuição uniforme
        for (int c = cfg; c < 4; c++) {
            cpt[i][c][0] = 0.5;
            cpt[i][c][1] = 0.5;
        }
    }
}

/* ================= SAMPLE ================= */

int sample(double *p) {
    // p é um array de 2 doubles [p(0), p(1)]
    double r = (double)rand() / RAND_MAX;
    return r < p[1] ? 1 : 0;
}

void sample_individual(Individual *ind) {
    for (int i = 0; i < N; i++) {
        int idx = 0;
        for (int p = 0; p < bn[i].num_parents; p++) {
            int parent = bn[i].parents[p];
            idx = (idx << 1) | ind->x[parent];
        }
        
        // Garante que idx está dentro dos limites
        int cfg = 1 << bn[i].num_parents;
        if (idx >= cfg) idx = cfg - 1;
        
        // Amostra o valor
        ind->x[i] = sample(cpt[i][idx]);
    }
    ind->fitness = fitness(ind);
}

/* ================= MAIN ================= */

int main() {
    srand(time(NULL));

    Individual pop[POP_MAX];
    int pop_size = INIT_POP;

    for (int i = 0; i < pop_size; i++)
        random_individual(&pop[i]);

    for (int g = 0; g < GENERATIONS; g++) {

        /* avaliação */
        for (int i = 0; i < pop_size; i++)
            pop[i].fitness = fitness(&pop[i]);

        qsort(pop, pop_size, sizeof(Individual), cmp);

        printf("Gen %3d | Best: %d | Pop: %d\n",
               g, pop[0].fitness, pop_size);
        
        if (pop[0].fitness == N) {
            printf("\nOptimal solution found at generation %d!\n", g);
            break;
        }

        /* preserva elite */
        Individual elite[ELITE_SIZE];
        int elite_count = 0;
        for (int i = 0; i < ELITE_SIZE && i < pop_size; i++) {
            elite[elite_count++] = pop[i];
        }

        /* aprende BN (apenas se população for suficiente) */
        if (pop_size >= 10) {
            learn_structure(pop, pop_size);
            learn_parameters(pop, pop_size);
        }

        /* amostra - gera novos indivíduos */
        int samples_to_generate = POP_MAX - pop_size;
        for (int i = 0; i < samples_to_generate; i++) {
            // Seleciona um indivíduo aleatório como base
            int idx = rand() % pop_size;
            Individual new_ind;
            // Copia o indivíduo base
            for (int j = 0; j < N; j++)
                new_ind.x[j] = pop[idx].x[j];
            // Amostra um novo indivíduo baseado na BN
            sample_individual(&new_ind);
            if (pop_size < POP_MAX) {
                pop[pop_size++] = new_ind;
            }
        }

        /* remove similares (mas mantém elite) */
        Individual temp[POP_MAX];
        int tsize = 0;
        
        // Primeiro adiciona elite
        for (int i = 0; i < elite_count; i++)
            temp[tsize++] = elite[i];
        
        // Depois adiciona outros que não são similares aos já adicionados
        for (int i = 0; i < pop_size; i++) {
            int already_in = 0;
            for (int j = 0; j < tsize; j++) {
                if (distance(&pop[i], &temp[j]) < SIM_THRESHOLD) {
                    already_in = 1;
                    break;
                }
            }
            if (!already_in && tsize < POP_MAX) {
                temp[tsize++] = pop[i];
            }
        }
        
        pop_size = tsize;
        for (int i = 0; i < pop_size; i++)
            pop[i] = temp[i];

        /* injeção aleatória */
        int inject = (int)(RANDOM_RATE * POP_MAX);
        for (int i = 0; i < inject && pop_size < POP_MAX; i++) {
            random_individual(&pop[pop_size]);
            pop_size++;
        }

        /* garante população mínima */
        if (pop_size < INIT_POP / 2) {
            for (int i = pop_size; i < INIT_POP; i++)
                random_individual(&pop[i]);
            pop_size = INIT_POP;
        }
    }

    /* avaliação final */
    for (int i = 0; i < pop_size; i++)
        pop[i].fitness = fitness(&pop[i]);
    qsort(pop, pop_size, sizeof(Individual), cmp);

    printf("\nBest solution:\n");
    for (int i = 0; i < N; i++)
        printf("%d ", pop[0].x[i]);

    printf("\nFitness = %d / %d\n", pop[0].fitness, N);

    return 0;
}