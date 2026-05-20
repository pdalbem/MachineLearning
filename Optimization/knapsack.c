#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* ================= PARÂMETROS ================= */

#define N 10
#define POP_MAX 200
#define INIT_POP 80
#define GENERATIONS 100

#define MAX_PARENTS 2
#define THRESHOLD 20        // Reduzido de 30 para preservar diversidade
#define SIM_THRESHOLD 3
#define RANDOM_RATE 0.05
#define ELITE_SIZE 5        // Adicionado elitismo

#define CAPACITY 35
#define PENALTY 10

/* ================= DADOS ================= */

int weight[N] = {7, 4, 8, 6, 5, 9, 3, 2, 6, 4};
int value[N]  = {10, 6, 12, 8, 7, 13, 4, 3, 9, 6};

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
double cpt[N][4][2];  // 4 configurações possíveis (2^2)

/* ================= FITNESS ================= */

int fitness(Individual *ind) {
    int w = 0, v = 0;

    for (int i = 0; i < N; i++) {
        if (ind->x[i]) {
            w += weight[i];
            v += value[i];
        }
    }

    if (w <= CAPACITY) return v;
    return v - PENALTY * (w - CAPACITY);
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

/* ================= DISTÂNCIA (HAMMING) ================= */

int hamming(Individual *a, Individual *b) {
    int d = 0;
    for (int i = 0; i < N; i++)
        if (a->x[i] != b->x[i]) d++;
    return d;
}

/* ================= BIC ================= */

double bic_score(Individual pop[], int size, int node, int parents[], int np) {
    int configs = 1 << np;
    int counts[4][2];
    
    // Laplace smoothing
    for (int i = 0; i < configs; i++)
        for (int j = 0; j < 2; j++)
            counts[i][j] = 1;

    for (int i = 0; i < size; i++) {
        int idx = 0;
        for (int p = 0; p < np; p++)
            idx = (idx << 1) | pop[i].x[parents[p]];
        if (idx < configs) {
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

    int k = configs;
    return loglik - 0.5 * k * log(size);
}

/* ================= ESTRUTURA CORRIGIDA ================= */

void learn_structure(Individual pop[], int size) {
    if (size < 10) return;  // População muito pequena
    
    for (int i = 0; i < N; i++) {
        bn[i].num_parents = 0;
        double best = bic_score(pop, size, i, NULL, 0);

        while (bn[i].num_parents < MAX_PARENTS) {
            int best_p = -1;

            // Permite qualquer nó como pai (exceto i mesmo)
            for (int p = 0; p < N; p++) {
                if (p == i) continue;
                
                int used = 0;
                for (int k = 0; k < bn[i].num_parents; k++)
                    if (bn[i].parents[k] == p) used = 1;
                if (used) continue;

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

            if (best_p >= 0) {
                bn[i].parents[bn[i].num_parents++] = best_p;
            } else {
                break;
            }
        }
    }
}

/* ================= PARÂMETROS CORRIGIDOS ================= */

void learn_parameters(Individual pop[], int size) {
    // Inicializa todas as 4 configurações para todos os nós
    for (int i = 0; i < N; i++) {
        // Inicializa com Laplace smoothing
        for (int c = 0; c < 4; c++) {
            cpt[i][c][0] = 1.0;
            cpt[i][c][1] = 1.0;
        }
        
        // Conta ocorrências
        for (int k = 0; k < size; k++) {
            int idx = 0;
            for (int p = 0; p < bn[i].num_parents; p++) {
                idx = (idx << 1) | pop[k].x[bn[i].parents[p]];
            }
            cpt[i][idx][pop[k].x[i]] += 1.0;
        }
        
        // Normaliza
        for (int c = 0; c < 4; c++) {
            double sum = cpt[i][c][0] + cpt[i][c][1];
            if (sum > 0) {
                cpt[i][c][0] /= sum;
                cpt[i][c][1] /= sum;
            } else {
                cpt[i][c][0] = 0.5;
                cpt[i][c][1] = 0.5;
            }
        }
    }
}

/* ================= SAMPLE COM PROTEÇÃO ================= */

int sample(double *p) {
    double r = (double)rand() / RAND_MAX;
    return r < p[1] ? 1 : 0;
}

void sample_individual(Individual *ind) {
    for (int i = 0; i < N; i++) {
        int idx = 0;
        for (int p = 0; p < bn[i].num_parents; p++) {
            idx = (idx << 1) | ind->x[bn[i].parents[p]];
        }
        // Garante que idx está entre 0 e 3
        idx = idx & 3;  // Máscara para 2 bits
        ind->x[i] = sample(cpt[i][idx]);
    }
    ind->fitness = fitness(ind);
}

/* ================= MAIN CORRIGIDO ================= */

int main() {
    srand(time(NULL));

    Individual pop[POP_MAX];
    int pop_size = INIT_POP;

    // Inicializa
    for (int i = 0; i < pop_size; i++)
        random_individual(&pop[i]);

    // Melhor fitness encontrado
    int best_fitness_ever = 0;
    Individual best_solution;

    for (int g = 0; g < GENERATIONS; g++) {

        // Avalia
        for (int i = 0; i < pop_size; i++)
            pop[i].fitness = fitness(&pop[i]);

        qsort(pop, pop_size, sizeof(Individual), cmp);

        // Atualiza melhor solução global
        if (pop[0].fitness > best_fitness_ever) {
            best_fitness_ever = pop[0].fitness;
            best_solution = pop[0];
        }

        printf("Gen %3d | Best: %d | Global Best: %d | Pop: %d\n",
               g, pop[0].fitness, best_fitness_ever, pop_size);

        // Preserva elite
        Individual elite[ELITE_SIZE];
        int elite_count = 0;
        for (int i = 0; i < ELITE_SIZE && i < pop_size; i++) {
            elite[elite_count++] = pop[i];
        }

        // Aprende BN
        if (pop_size >= 10) {
            learn_structure(pop, pop_size);
            learn_parameters(pop, pop_size);
        }

        // Amostragem
        int new_size = pop_size;
        int samples = POP_MAX - pop_size;
        for (int i = 0; i < samples; i++) {
            // Seleciona um indivíduo aleatório como base
            int idx = rand() % pop_size;
            Individual new_ind;
            for (int j = 0; j < N; j++)
                new_ind.x[j] = pop[idx].x[j];
            sample_individual(&new_ind);
            if (new_size < POP_MAX) {
                pop[new_size++] = new_ind;
            }
        }
        pop_size = new_size;

        // Filtro de fitness (mais brando)
        int k = 0;
        for (int i = 0; i < pop_size; i++) {
            if (pop[i].fitness >= THRESHOLD || i < ELITE_SIZE) {
                pop[k++] = pop[i];
            }
        }
        pop_size = k;

        // Remove similares (preservando elite)
        Individual temp[POP_MAX];
        int tsize = 0;
        
        // Primeiro adiciona elite
        for (int i = 0; i < elite_count; i++)
            temp[tsize++] = elite[i];
        
        // Depois adiciona outros não similares
        for (int i = 0; i < pop_size; i++) {
            int keep = 1;
            for (int j = 0; j < tsize; j++) {
                if (hamming(&pop[i], &temp[j]) < SIM_THRESHOLD) {
                    keep = 0;
                    break;
                }
            }
            if (keep && tsize < POP_MAX) {
                temp[tsize++] = pop[i];
            }
        }
        
        pop_size = tsize;
        for (int i = 0; i < pop_size; i++)
            pop[i] = temp[i];

        // Injeção aleatória
        int inject = (int)(RANDOM_RATE * POP_MAX);
        for (int i = 0; i < inject && pop_size < POP_MAX; i++) {
            random_individual(&pop[pop_size]);
            pop_size++;
        }

        // Evita colapso
        if (pop_size < INIT_POP / 2) {
            for (int i = pop_size; i < INIT_POP; i++)
                random_individual(&pop[i]);
            pop_size = INIT_POP;
        }
    }

    // Avaliação final
    for (int i = 0; i < pop_size; i++)
        pop[i].fitness = fitness(&pop[i]);
    qsort(pop, pop_size, sizeof(Individual), cmp);
    
    if (pop[0].fitness > best_fitness_ever) {
        best_fitness_ever = pop[0].fitness;
        best_solution = pop[0];
    }

    printf("\nBest solution found:\n");
    printf("Items selected: ");
    int total_weight = 0, total_value = 0;
    for (int i = 0; i < N; i++) {
        printf("%d ", best_solution.x[i]);
        if (best_solution.x[i]) {
            total_weight += weight[i];
            total_value += value[i];
        }
    }
    
    printf("\nTotal weight: %d / %d", total_weight, CAPACITY);
    printf("\nTotal value: %d", total_value);
    printf("\nFitness = %d\n", best_fitness_ever);

    return 0;
}