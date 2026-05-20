#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* ================= PARÂMETROS ================= */

#define N 10
#define POP_MAX 200
#define INIT_POP 80
#define GENERATIONS 100

#define SIM_THRESHOLD 3
#define RANDOM_RATE 0.05

/* ================= MATRIZES ================= */

double dist[N][N];
double cost[N][N];

/* ================= ESTRUTURAS ================= */

typedef struct {
    int tour[N];
    double f1, f2;
} Individual;

/* ================= GERAÇÃO ================= */

void generate_problem() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                dist[i][j] = 0;
                cost[i][j] = 0;
            } else {
                dist[i][j] = rand() % 100 + 1;
                cost[i][j] = rand() % 100 + 1;
            }
        }
    }
}

/* ================= AVALIAÇÃO ================= */

void evaluate(Individual *ind) {
    double d = 0, c = 0;

    for (int i = 0; i < N - 1; i++) {
        int a = ind->tour[i];
        int b = ind->tour[i + 1];
        d += dist[a][b];
        c += cost[a][b];
    }

    // retorno ao início
    d += dist[ind->tour[N-1]][ind->tour[0]];
    c += cost[ind->tour[N-1]][ind->tour[0]];

    ind->f1 = d;
    ind->f2 = c;
}

/* ================= DOMINÂNCIA ================= */

int dominates(Individual *a, Individual *b) {
    return (a->f1 <= b->f1 && a->f2 <= b->f2) &&
           (a->f1 < b->f1 || a->f2 < b->f2);
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
    int used[N] = {0};

    for (int i = 0; i < N; i++) {
        int r;
        do {
            r = rand() % N;
        } while (used[r]);

        ind->tour[i] = r;
        used[r] = 1;
    }

    evaluate(ind);
}

/* distância entre soluções */
int distance_perm(Individual *a, Individual *b) {
    int d = 0;
    for (int i = 0; i < N; i++)
        if (a->tour[i] != b->tour[i]) d++;
    return d;
}

/* ================= MODELO (EDA) ================= */

double prob[N][N]; // prob[i][j] = prob de ir i -> j

void learn_model(Individual pop[], int size) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            prob[i][j] = 1.0;

    for (int k = 0; k < size; k++) {
        for (int i = 0; i < N - 1; i++) {
            int a = pop[k].tour[i];
            int b = pop[k].tour[i+1];
            prob[a][b]++;
        }
    }

    for (int i = 0; i < N; i++) {
        double sum = 0;
        for (int j = 0; j < N; j++)
            sum += prob[i][j];

        for (int j = 0; j < N; j++)
            prob[i][j] /= sum;
    }
}

/* ================= AMOSTRAGEM ================= */

int sample_next(int current, int used[]) {
    double r = (double)rand() / RAND_MAX;
    double acc = 0;

    for (int j = 0; j < N; j++) {
        if (used[j]) continue;

        acc += prob[current][j];
        if (r <= acc)
            return j;
    }

    for (int j = 0; j < N; j++)
        if (!used[j]) return j;

    return 0;
}

void sample_individual(Individual *ind) {
    int used[N] = {0};

    ind->tour[0] = rand() % N;
    used[ind->tour[0]] = 1;

    for (int i = 1; i < N; i++) {
        int prev = ind->tour[i-1];
        int next = sample_next(prev, used);
        ind->tour[i] = next;
        used[next] = 1;
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

        printf("Gen %3d | Pareto: %d | Pop: %d\n",
               g, front_size, pop_size);

        learn_model(front, front_size);

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
                if (distance_perm(&pop[i], &temp[j]) < SIM_THRESHOLD) {
                    keep = 0;
                    break;
                }
            }
            if (keep) temp[tsize++] = pop[i];
        }

        for (int i = 0; i < tsize; i++)
            pop[i] = temp[i];
        pop_size = tsize;

        /* aleatórios */
        int inject = (int)(RANDOM_RATE * pop_size);

        for (int i = 0; i < inject && pop_size < POP_MAX; i++) {
            random_individual(&pop[pop_size]);
            pop_size++;
        }
    }

    /* resultado */
    Individual front[POP_MAX];
    int front_size = get_pareto_front(pop, pop_size, front);

    printf("\nPareto front:\n");
    for (int i = 0; i < front_size; i++) {
        printf("Dist=%.2f Cost=%.2f\n",
               front[i].f1, front[i].f2);
    }

    return 0;
}