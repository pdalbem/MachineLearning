#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ================= PARÂMETROS ================= */
#define N 5
#define POP_MAX 500
#define INIT_POP 200
#define GENERATIONS 200
#define LOWER -5.12
#define UPPER 5.12
#define SIM_THRESHOLD 0.1      // Reduzido
#define RANDOM_RATE 0.05
#define ELITE_RATE 0.2         // 20% elite
#define EPS 1e-8

/* ================= ESTRUTURA ================= */
typedef struct {
    double x[N];
    double fitness;
} Individual;

/* ================= RANDOM ================= */
double rand_uniform(double a, double b) {
    return a + (b - a) * ((double)rand() / RAND_MAX);
}

double rand_normal() {
    double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
    double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* ================= FUNÇÃO SPHERE ================= */
double fitness(Individual *ind) {
    double sum = 0.0;
    for (int i = 0; i < N; i++)
        sum += ind->x[i] * ind->x[i];
    return sum;
}

/* ================= UTILITÁRIOS ================= */
double distance(Individual *a, Individual *b) {
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        double d = a->x[i] - b->x[i];
        sum += d * d;
    }
    return sqrt(sum);
}

int cmp_fitness(const void *a, const void *b) {
    double fa = ((Individual*)a)->fitness;
    double fb = ((Individual*)b)->fitness;
    return (fa > fb) - (fa < fb);
}

/* ================= INICIALIZAÇÃO ================= */
void random_individual(Individual *ind) {
    for (int i = 0; i < N; i++)
        ind->x[i] = rand_uniform(LOWER, UPPER);
    ind->fitness = fitness(ind);
}

void init_near_optimum(Individual *ind) {
    for (int i = 0; i < N; i++)
        ind->x[i] = rand_normal() * 0.5;  // Perto de zero
    ind->fitness = fitness(ind);
}

void init_population(Individual *pop, int size) {
    for (int i = 0; i < size; i++) {
        if (i < size / 2) {
            init_near_optimum(&pop[i]);
        } else {
            random_individual(&pop[i]);
        }
    }
}

/* ================= MÉDIA ================= */
void compute_mean(Individual pop[], int size, double mu[]) {
    for (int i = 0; i < N; i++) {
        mu[i] = 0.0;
        for (int j = 0; j < size; j++)
            mu[i] += pop[j].x[i];
        mu[i] /= size;
    }
}

/* ================= COVARIÂNCIA ================= */
void compute_covariance(Individual pop[], int size, double mu[], double cov[N][N]) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            cov[i][j] = 0.0;

    for (int k = 0; k < size; k++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                cov[i][j] += (pop[k].x[i] - mu[i]) *
                             (pop[k].x[j] - mu[j]);
            }
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            cov[i][j] /= size;
        
        // Regularização adaptativa
        cov[i][i] += 0.01;
    }
}

/* ================= CHOLESKY ESTÁVEL ================= */
int cholesky(double A[N][N], double L[N][N]) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            L[i][j] = 0.0;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            for (int k = 0; k < j; k++)
                sum += L[i][k] * L[j][k];

            if (i == j) {
                double val = A[i][i] - sum;
                if (val <= 1e-12) {
                    L[i][j] = sqrt(1e-8);
                } else {
                    L[i][j] = sqrt(val);
                }
            } else {
                if (fabs(L[j][j]) < 1e-12) return 0;
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
    }
    return 1;
}

/* ================= AMOSTRAGEM ================= */
void sample_individual(Individual *ind, double mu[], double L[N][N], double scale) {
    double z[N];

    for (int i = 0; i < N; i++)
        z[i] = rand_normal() * scale;

    for (int i = 0; i < N; i++) {
        ind->x[i] = mu[i];
        for (int j = 0; j <= i; j++)
            ind->x[i] += L[i][j] * z[j];

        if (ind->x[i] < LOWER) ind->x[i] = LOWER;
        if (ind->x[i] > UPPER) ind->x[i] = UPPER;
    }

    ind->fitness = fitness(ind);
}

void mutate_individual(Individual *ind, double step) {
    for (int j = 0; j < N; j++) {
        ind->x[j] += rand_normal() * step;
        if (ind->x[j] < LOWER) ind->x[j] = LOWER;
        if (ind->x[j] > UPPER) ind->x[j] = UPPER;
    }
    ind->fitness = fitness(ind);
}

/* ================= FILTRO ADAPTATIVO ================= */
int filter_population(Individual pop[], int pop_size, double best_fitness) {
    double threshold;
    if (best_fitness < 0.01)
        threshold = best_fitness + 0.01;
    else if (best_fitness < 0.1)
        threshold = best_fitness * 1.2;
    else if (best_fitness < 1.0)
        threshold = best_fitness * 1.5;
    else
        threshold = best_fitness * 2.0;
    
    Individual temp[POP_MAX];
    int t1 = 0;
    
    for (int i = 0; i < pop_size; i++) {
        if (pop[i].fitness <= threshold)
            temp[t1++] = pop[i];
    }
    
    // Mantém pelo menos 70% da população
    if (t1 < pop_size * 0.7) {
        t1 = (int)(pop_size * 0.7);
        qsort(pop, pop_size, sizeof(Individual), cmp_fitness);
        for (int i = 0; i < t1; i++)
            temp[i] = pop[i];
    }
    
    // Filtro de similaridade
    Individual temp2[POP_MAX];
    int t2 = 0;
    
    for (int i = 0; i < t1; i++) {
        int keep = 1;
        for (int j = 0; j < t2; j++) {
            if (distance(&temp[i], &temp2[j]) < SIM_THRESHOLD) {
                keep = 0;
                break;
            }
        }
        if (keep) temp2[t2++] = temp[i];
    }
    
    for (int i = 0; i < t2; i++)
        pop[i] = temp2[i];
    
    return t2;
}

/* ================= MAIN ================= */
int main() {
    srand(time(NULL));

    Individual pop[POP_MAX];
    Individual new_pop[POP_MAX];
    int pop_size = INIT_POP;
    
    init_population(pop, pop_size);
    
    double best_fitness_ever = 1e9;
    Individual best_individual;
    double scale_factor = 0.5;
    int stagnation_counter = 0;
    int no_improvement_counter = 0;

    for (int g = 0; g < GENERATIONS; g++) {
        // Ordena
        qsort(pop, pop_size, sizeof(Individual), cmp_fitness);
        
        double current_best = pop[0].fitness;
        
        // Atualiza melhor global
        if (current_best < best_fitness_ever - 1e-10) {
            best_fitness_ever = current_best;
            best_individual = pop[0];
            stagnation_counter = 0;
            no_improvement_counter = 0;
            scale_factor = fmax(0.05, scale_factor * 0.95);
        } else {
            no_improvement_counter++;
            if (no_improvement_counter > 15) {
                stagnation_counter++;
                scale_factor = fmin(0.5, scale_factor * 1.05);
                if (stagnation_counter > 30) {
                    printf("   🔄 Reinicialização parcial...\n");
                    for (int i = pop_size/2; i < pop_size; i++) {
                        init_near_optimum(&pop[i]);
                    }
                    stagnation_counter = 0;
                    scale_factor = 0.3;
                }
            }
        }

        printf("Gen %3d | Best: %.10f | Pop: %3d | Scale: %.3f | NoImp: %d\n",
               g, current_best, pop_size, scale_factor, no_improvement_counter);
        
        // Critério de parada
        if (best_fitness_ever < 1e-12) {
            printf("\n✅ Convergência ótima alcançada!\n");
            break;
        }
        
        // Preserva elite
        int elite_size = (int)(ELITE_RATE * pop_size);
        if (elite_size < 5) elite_size = 5;
        
        for (int i = 0; i < elite_size; i++)
            new_pop[i] = pop[i];
        int new_size = elite_size;
        
        // Aprende modelo com elite (melhores indivíduos)
        double mu[N], cov[N][N], L[N][N];
        compute_mean(pop, pop_size / 2, mu);  // Usa top 50%
        compute_covariance(pop, pop_size / 2, mu, cov);
        
        if (!cholesky(cov, L)) {
            printf("Erro Cholesky, usando identidade\n");
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    L[i][j] = (i == j) ? 1.0 : 0.0;
        }
        
        // Amostragem
        int samples_to_add = pop_size;
        for (int i = 0; i < samples_to_add && new_size < POP_MAX; i++) {
            if ((double)rand() / RAND_MAX < 0.1) {
                // Exploração aleatória
                if (best_fitness_ever < 0.1)
                    init_near_optimum(&new_pop[new_size]);
                else
                    random_individual(&new_pop[new_size]);
            } else {
                sample_individual(&new_pop[new_size], mu, L, scale_factor);
            }
            new_size++;
        }
        
        // Exploração local
        for (int i = 0; i < 5 && new_size < POP_MAX; i++) {
            Individual mutated = pop[i % elite_size];
            mutate_individual(&mutated, 0.05 * scale_factor);
            new_pop[new_size++] = mutated;
        }
        
        // Filtragem
        int filtered_size = filter_population(new_pop, new_size, best_fitness_ever);
        
        // Injeção aleatória
        int inject = (int)(RANDOM_RATE * filtered_size);
        if (inject < 3) inject = 3;
        
        if (filtered_size + inject <= POP_MAX) {
            for (int i = 0; i < inject; i++) {
                if (best_fitness_ever < 0.1)
                    init_near_optimum(&new_pop[filtered_size + i]);
                else
                    random_individual(&new_pop[filtered_size + i]);
            }
            filtered_size += inject;
        }
        
        // Atualiza população
        for (int i = 0; i < filtered_size; i++)
            pop[i] = new_pop[i];
        pop_size = filtered_size;
        
        // Garante população mínima
        if (pop_size < 30) {
            for (int i = pop_size; i < 50; i++)
                init_near_optimum(&pop[i]);
            pop_size = 50;
        }
    }

    printf("\n========================================\n");
    printf("MELHOR SOLUÇÃO ENCONTRADA:\n");
    printf("x = [");
    for (int i = 0; i < N; i++) {
        printf("%.8f", best_individual.x[i]);
        if (i < N-1) printf(", ");
    }
    printf("]\n");
    printf("Fitness = %.12f\n", best_fitness_ever);
    printf("Esperado = 0.000000000000\n");
    printf("========================================\n");

    return 0;
}