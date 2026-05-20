#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* ================= PARÂMETROS ================= */

#define N 12
#define POP_MAX 200
#define INIT_POP 80
#define GENERATIONS 100

#define CAPACITY 40

#define MAX_PARENTS 2
#define SIM_THRESHOLD 3
#define RANDOM_RATE 0.05

/* ================= DADOS ================= */

int weight[N];
int value1[N];
int value2[N];

/* ================= ESTRUTURAS ================= */

typedef struct {
    int x[N];
    int f1, f2;
    int weight;
} Individual;

typedef struct {
    int parents[MAX_PARENTS];
    int num_parents;
} Node;

/* ================= GLOBAIS ================= */

Node bn[N];
double cpt[N][4][2];

/* ================= GERAÇÃO ALEATÓRIA ================= */

void generate_problem() {
    for (int i = 0; i < N; i++) {
        weight[i] = rand() % 10 + 1;   // 1..10
        value1[i] = rand() % 20 + 1;   // 1..20
        value2[i] = rand() % 20 + 1;   // 1..20
    }

    printf("Itens:\n");
    for (int i = 0; i < N; i++) {
        printf("Item %2d | w=%2d v1=%2d v2=%2d\n",
               i, weight[i], value1[i], value2[i]);
    }
    printf("\n");
}

/* ================= AVALIAÇÃO ================= */

void evaluate(Individual *ind) {
    int w = 0, v1 = 0, v2 = 0;

    for (int i = 0; i < N; i++) {
        w += ind->x[i] * weight[i];
        v1 += ind->x[i] * value1[i];
        v2 += ind->x[i] * value2[i];
    }

    ind->weight = w;

    if (w > CAPACITY) {
        ind->f1 = 0;
        ind->f2 = 0;
    } else {
        ind->f1 = v1;
        ind->f2 = v2;
    }
}

/* ================= DOMINÂNCIA ================= */

int dominates(Individual *a, Individual *b) {
    return (a->f1 >= b->f1 && a->f2 >= b->f2) &&
           (a->f1 > b->f1 || a->f2 > b->f2);
}

/* ================= PARETO ================= */

int get_pareto_front(Individual pop[], int size, Individual front[]) {
    int count = 0;

    for (int i = 0; i < size; i++) {
        int dominated_flag = 0;

        for (int j = 0; j < size; j++) {
            if (j != i && dominates(&pop[j], &pop[i])) {
                dominated_flag = 1;
                break;
            }
        }

        if (!dominated_flag)
            front[count++] = pop[i];
    }

    return count;
}

/* ================= UTIL ================= */

void random_individual(Individual *ind) {
    for (int i = 0; i < N; i++)
        ind->x[i] = rand() % 2;
    evaluate(ind);
}

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

    for (int i = 0; i < configs; i++)
        for (int j = 0; j < 2; j++)
            counts[i][j] = 1;

    for (int i = 0; i < size; i++) {
        int idx = 0;
        for (int p = 0; p < np; p++)
            idx = (idx << 1) | pop[i].x[parents[p]];
        counts[idx][pop[i].x[node]]++;
    }

    double loglik = 0;
    for (int i = 0; i < configs; i++) {
        int sum = counts[i][0] + counts[i][1];
        for (int j = 0; j < 2; j++)
            loglik += counts[i][j] * log((double)counts[i][j] / sum);
    }

    int k = configs;
    return loglik - 0.5 * k * log(size);
}

/* ================= BN ================= */

void learn_structure(Individual pop[], int size) {
    for (int i = 0; i < N; i++) {
        bn[i].num_parents = 0;
        double best = bic_score(pop, size, i, NULL, 0);

        while (bn[i].num_parents < MAX_PARENTS) {
            int best_p = -1;

            for (int p = 0; p < i; p++) {
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

            if (best_p >= 0)
                bn[i].parents[bn[i].num_parents++] = best_p;
            else break;
        }
    }
}

void learn_parameters(Individual pop[], int size) {
    for (int i = 0; i < N; i++) {
        int cfg = 1 << bn[i].num_parents;

        for (int c = 0; c < cfg; c++)
            for (int v = 0; v < 2; v++)
                cpt[i][c][v] = 1.0;

        for (int k = 0; k < size; k++) {
            int idx = 0;
            for (int p = 0; p < bn[i].num_parents; p++)
                idx = (idx << 1) | pop[k].x[bn[i].parents[p]];
            cpt[i][idx][pop[k].x[i]]++;
        }

        for (int c = 0; c < cfg; c++) {
            double sum = cpt[i][c][0] + cpt[i][c][1];
            cpt[i][c][0] /= sum;
            cpt[i][c][1] /= sum;
        }
    }
}

/* ================= SAMPLE ================= */

int sample(double *p) {
    return ((double)rand() / RAND_MAX) < p[1];
}

void sample_individual(Individual *ind) {
    for (int i = 0; i < N; i++) {
        int idx = 0;
        for (int p = 0; p < bn[i].num_parents; p++)
            idx = (idx << 1) | ind->x[bn[i].parents[p]];

        ind->x[i] = sample(cpt[i][idx]);
    }
    evaluate(ind);
}

/* ================= MAIN ================= */

int main() {
    srand(time(NULL));

    generate_problem();

    Individual pop[POP_MAX];
    int pop_size = INIT_POP;

    for (int i = 0; i < pop_size; i++)
        random_individual(&pop[i]);

    for (int g = 0; g < GENERATIONS; g++) {

        Individual front[POP_MAX];
        int front_size = get_pareto_front(pop, pop_size, front);

        printf("Gen %3d | Pareto size: %d | Pop: %d\n",
               g, front_size, pop_size);

        learn_structure(front, front_size);
        learn_parameters(front, front_size);

        int new_size = pop_size;

        for (int i = 0; i < front_size && new_size < POP_MAX; i++) {
            sample_individual(&pop[new_size]);
            new_size++;
        }
        pop_size = new_size;

        /* diversidade */
        Individual temp[POP_MAX];
        int tsize = 0;

        for (int i = 0; i < pop_size; i++) {
            int keep = 1;
            for (int j = 0; j < tsize; j++) {
                if (hamming(&pop[i], &temp[j]) < SIM_THRESHOLD) {
                    keep = 0;
                    break;
                }
            }
            if (keep) temp[tsize++] = pop[i];
        }

        for (int i = 0; i < tsize; i++)
            pop[i] = temp[i];
        pop_size = tsize;

        /* injeção */
        int inject = (int)(RANDOM_RATE * pop_size);

        for (int i = 0; i < inject && pop_size < POP_MAX; i++) {
            random_individual(&pop[pop_size]);
            pop_size++;
        }
    }

    /* saída final */
    Individual front[POP_MAX];
    int front_size = get_pareto_front(pop, pop_size, front);

    printf("\nPareto front:\n");
    for (int i = 0; i < front_size; i++)
        printf("f1=%d f2=%d w=%d\n",
               front[i].f1, front[i].f2, front[i].weight);

    return 0;
}