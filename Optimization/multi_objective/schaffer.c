#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define D 1                 // Dimensão do problema (Schaffer F1/F2)
#define INIT_POP 50         // População inicial
#define MAX_GEN 100
#define MAX_CLUSTERS 5
#define SIMILARITY_THRESHOLD 0.01
#define NEW_INDIVIDUALS_RATIO 0.05

typedef struct {
    double x[D];
    double f1, f2;
} Individual;

typedef struct {
    Individual *inds;
    int size;
} Population;

/* ================= FUNÇÃO DE FITNESS ================= */
void evaluate(Individual *ind) {
    double xi = ind->x[0];
    ind->f1 = xi * xi;
    ind->f2 = (xi - 2.0) * (xi - 2.0);
}

/* ================= DISTÂNCIA EUCLIDIANA ================= */
double distance(Individual *a, Individual *b) {
    double sum = 0;
    for (int i = 0; i < D; i++) sum += (a->x[i] - b->x[i]) * (a->x[i] - b->x[i]);
    return sqrt(sum);
}

/* ================= INICIALIZAÇÃO ================= */
void init_population(Population *pop, int size) {
    pop->size = size;
    pop->inds = (Individual *)malloc(size * sizeof(Individual));
    for (int i = 0; i < size; i++) {
        for (int d = 0; d < D; d++)
            pop->inds[i].x[d] = ((double)rand() / RAND_MAX) * 4.0 - 2.0; // [-2,2]
        evaluate(&pop->inds[i]);
    }
}

/* ================= ELIMINAÇÃO DE SOLUÇÕES SIMILARES ================= */
void remove_similar(Population *pop) {
    for (int i = 0; i < pop->size; i++) {
        for (int j = i + 1; j < pop->size; ) {
            if (distance(&pop->inds[i], &pop->inds[j]) < SIMILARITY_THRESHOLD) {
                for (int k = j; k < pop->size - 1; k++)
                    pop->inds[k] = pop->inds[k + 1];
                pop->size--;
            } else j++;
        }
    }
}

/* ================= FRONT DE PARETO ================= */
int dominates(Individual *a, Individual *b) {
    return ((a->f1 <= b->f1 && a->f2 <= b->f2) && (a->f1 < b->f1 || a->f2 < b->f2));
}

void pareto_filter(Population *pop) {
    int i = 0;
    while (i < pop->size) {
        int dominated = 0;
        for (int j = 0; j < pop->size; j++) {
            if (i != j && dominates(&pop->inds[j], &pop->inds[i])) {
                dominated = 1; break;
            }
        }
        if (dominated) {
            for (int k = i; k < pop->size - 1; k++) pop->inds[k] = pop->inds[k + 1];
            pop->size--;
        } else i++;
    }
}

/* ================= CLUSTERING KMEANS ================= */
void kmeans(Individual *inds, int n, int clusters, double centers[clusters][D], int assignments[n]) {
    // Inicializa centros aleatórios
    for (int c = 0; c < clusters; c++) {
        int idx = rand() % n;
        for (int d = 0; d < D; d++) centers[c][d] = inds[idx].x[d];
    }

    int changed = 1;
    while (changed) {
        changed = 0;
        for (int i = 0; i < n; i++) {
            int best = 0;
            double best_dist = distance(&inds[i], &(Individual){.x = {centers[0][0]}});
            for (int c = 1; c < clusters; c++) {
                double dist = 0;
                for (int d = 0; d < D; d++) dist += (inds[i].x[d] - centers[c][d]) * (inds[i].x[d] - centers[c][d]);
                dist = sqrt(dist);
                if (dist < best_dist) { best_dist = dist; best = c; }
            }
            if (assignments[i] != best) { assignments[i] = best; changed = 1; }
        }
        // Atualiza centros
        for (int c = 0; c < clusters; c++) {
            double sum[D] = {0};
            int count = 0;
            for (int i = 0; i < n; i++)
                if (assignments[i] == c) { for (int d = 0; d < D; d++) sum[d] += inds[i].x[d]; count++; }
            if (count > 0) for (int d = 0; d < D; d++) centers[c][d] = sum[d] / count;
        }
    }
}

/* ================= AMOSTRAGEM GMM ================= */
void sample_gmm(Population *pop, Individual *out) {
    int clusters = (pop->size < MAX_CLUSTERS) ? pop->size : MAX_CLUSTERS;
    int assignments[pop->size];
    double centers[clusters][D];
    kmeans(pop->inds, pop->size, clusters, centers, assignments);

    // Seleciona cluster aleatório
    int c = rand() % clusters;

    // Calcula desvio padrão do cluster
    double stddev[D] = {0};
    int count = 0;
    for (int i = 0; i < pop->size; i++)
        if (assignments[i] == c) {
            for (int d = 0; d < D; d++)
                stddev[d] += (pop->inds[i].x[d] - centers[c][d]) * (pop->inds[i].x[d] - centers[c][d]);
            count++;
        }
    if (count > 0) for (int d = 0; d < D; d++) stddev[d] = sqrt(stddev[d] / count);

    // Amostra nova solução com ruído adaptativo
    for (int d = 0; d < D; d++)
        out->x[d] = centers[c][d] + stddev[d] * ((double)rand() / RAND_MAX - 0.5) * 2.0;

    evaluate(out);
}

/* ================= INSERE NOVOS INDIVÍDUOS ALEATÓRIOS ================= */
void insert_new(Population *pop) {
    int n_new = (int)(pop->size * NEW_INDIVIDUALS_RATIO);
    pop->inds = (Individual *)realloc(pop->inds, (pop->size + n_new) * sizeof(Individual));
    for (int i = 0; i < n_new; i++) {
        for (int d = 0; d < D; d++)
            pop->inds[pop->size + i].x[d] = ((double)rand() / RAND_MAX) * 4.0 - 2.0;
        evaluate(&pop->inds[pop->size + i]);
    }
    pop->size += n_new;
}

/* ================= MAIN ================= */
int main() {
    srand(time(NULL));

    Population pop;
    init_population(&pop, INIT_POP);

    for (int gen = 0; gen < MAX_GEN; gen++) {
        // Mantém Pareto front
        pareto_filter(&pop);

        // Remove soluções similares
        remove_similar(&pop);

        // Amostra novos indivíduos via GMM
        int orig_size = pop.size;
        for (int i = 0; i < orig_size; i++) {
            Individual new_ind;
            sample_gmm(&pop, &new_ind);
            pop.inds = (Individual *)realloc(pop.inds, (pop.size + 1) * sizeof(Individual));
            pop.inds[pop.size++] = new_ind;
        }

        // Insere novos aleatórios
        insert_new(&pop);

        // Controle adaptativo do tamanho da população
        if (pop.size < 20) {
            int extra = 20 - pop.size;
            pop.inds = (Individual *)realloc(pop.inds, (pop.size + extra) * sizeof(Individual));
            for (int i = 0; i < extra; i++) {
                pop.inds[pop.size + i].x[0] = ((double)rand() / RAND_MAX) * 4.0 - 2.0;
                evaluate(&pop.inds[pop.size + i]);
            }
            pop.size += extra;
        }

        printf("Gen %3d | Pareto size: %3d | Pop: %3d\n", gen, pop.size, pop.size);
    }

    // Imprime Pareto front
    printf("\nPareto front:\n");
    for (int i = 0; i < pop.size; i++)
        printf("x=%.6f f1=%.6f f2=%.6f\n", pop.inds[i].x[0], pop.inds[i].f1, pop.inds[i].f2);

    free(pop.inds);
    return 0;
}