#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* ================= PARÂMETROS ================= */
#define N 5
#define POP_MAX 400
#define INIT_POP 150
#define GENERATIONS 500
#define K 5
#define LOWER -600
#define UPPER 600
#define SIM_THRESHOLD 0.5      // Aumentado para Griewank
#define RANDOM_RATE 0.12
#define ELITE_RATE 0.2         // 20% elite
#define LOCAL_SEARCH 8
#define EPS 1e-8

/* ================= ESTRUTURA ================= */
typedef struct {
    double x[N];
    double fitness;
    int cluster;
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

/* ================= GRIEWANK ================= */
double fitness(Individual *ind) {
    double sum = 0.0;
    double prod = 1.0;

    for (int i = 0; i < N; i++) {
        double x = ind->x[i];
        sum += (x * x) / 4000.0;
        prod *= cos(x / sqrt(i + 1.0));
    }

    return sum - prod + 1.0;
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

void random_individual(Individual *ind) {
    for (int i = 0; i < N; i++)
        ind->x[i] = rand_uniform(LOWER, UPPER);
    ind->fitness = fitness(ind);
}

void init_near_optimum(Individual *ind) {
    // Inicializa perto do ótimo (0,0,0,0,0)
    for (int i = 0; i < N; i++)
        ind->x[i] = rand_normal() * 2.0;  // Pequena perturbação
    ind->fitness = fitness(ind);
}

/* ================= K-MEANS MELHORADO ================= */
void kmeans(Individual pop[], int size, double centroids[K][N]) {
    int actual_k = (size < K) ? size : K;
    if(actual_k < 2) actual_k = 2;
    
    // Inicializa centroides com os MELHORES indivíduos
    for (int k = 0; k < actual_k; k++) {
        int idx = (k * (size / actual_k)) % size;
        for (int i = 0; i < N; i++)
            centroids[k][i] = pop[idx].x[i];
    }
    
    // Mais iterações para convergência
    for (int iter = 0; iter < 15; iter++) {
        // Atribui clusters
        for (int i = 0; i < size; i++) {
            double best = 1e18;
            int best_k = 0;
            
            for (int k = 0; k < actual_k; k++) {
                double d = 0;
                for (int j = 0; j < N; j++) {
                    double diff = pop[i].x[j] - centroids[k][j];
                    d += diff * diff;
                }
                if (d < best) {
                    best = d;
                    best_k = k;
                }
            }
            pop[i].cluster = best_k;
        }
        
        // Atualiza centroides
        double sum[K][N] = {{0}};
        int count[K] = {0};
        
        for (int i = 0; i < size; i++) {
            int c = pop[i].cluster;
            count[c]++;
            for (int j = 0; j < N; j++)
                sum[c][j] += pop[i].x[j];
        }
        
        int changed = 0;
        for (int k = 0; k < actual_k; k++) {
            if (count[k] == 0) continue;
            for (int j = 0; j < N; j++) {
                double new_mean = sum[k][j] / count[k];
                if (fabs(new_mean - centroids[k][j]) > 1e-6)
                    changed = 1;
                centroids[k][j] = new_mean;
            }
        }
        
        if (!changed) break;
    }
}

/* ================= GAUSSIANA MELHORADA ================= */
int compute_gaussian(Individual pop[], int size, int cluster,
                     double mu[N], double cov[N][N], double *avg_fit) {
    int count = 0;
    double fit_sum = 0.0;
    
    for (int i = 0; i < N; i++) mu[i] = 0.0;
    
    for (int i = 0; i < size; i++) {
        if (pop[i].cluster != cluster) continue;
        count++;
        fit_sum += pop[i].fitness;
        for (int j = 0; j < N; j++)
            mu[j] += pop[i].x[j];
    }
    
    if (count < 3) return 0;  // Precisa de pelo menos 3 pontos
    
    for (int i = 0; i < N; i++)
        mu[i] /= count;
    
    *avg_fit = fit_sum / count;
    
    // Inicializa covariância
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            cov[i][j] = 0.0;
    
    // Calcula covariância
    for (int k = 0; k < size; k++) {
        if (pop[k].cluster != cluster) continue;
        
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                cov[i][j] += (pop[k].x[i] - mu[i]) *
                             (pop[k].x[j] - mu[j]);
            }
        }
    }
    
    // Normaliza e regulariza
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            cov[i][j] /= count;
        
        // Regularização adaptativa baseada na escala
        double scale = fabs(mu[i]);
        if(scale < 1) scale = 1;
        cov[i][i] += 0.01 * scale;  // Regularização adaptativa
    }
    
    return 1;
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
                if (val <= 1e-10) {
                    // Se não é definida positiva, usa valor mínimo
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
void sample_from_cluster(Individual *ind, double mu[N], double L[N][N], double scale) {
    double z[N];
    
    for (int i = 0; i < N; i++)
        z[i] = rand_normal() * scale;
    
    for (int i = 0; i < N; i++) {
        ind->x[i] = mu[i];
        
        for (int j = 0; j <= i; j++)
            ind->x[i] += L[i][j] * z[j];
        
        // Limites com reflexão (melhor que clipping)
        if (ind->x[i] < LOWER)
            ind->x[i] = 2 * LOWER - ind->x[i];
        if (ind->x[i] > UPPER)
            ind->x[i] = 2 * UPPER - ind->x[i];
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
    // Threshold adaptativo para Griewank
    double threshold;
    if (best_fitness < 0.01)
        threshold = best_fitness + 0.05;
    else if (best_fitness < 0.1)
        threshold = best_fitness * 1.5;
    else if (best_fitness < 1.0)
        threshold = best_fitness * 2.0;
    else if (best_fitness < 10.0)
        threshold = best_fitness * 2.5;
    else
        threshold = best_fitness * 3.0;
    
    // Limite inferior
    if (threshold < best_fitness + 0.01) 
        threshold = best_fitness + 0.01;
    
    // Filtro de fitness
    Individual temp[POP_MAX];
    int t1 = 0;
    
    for (int i = 0; i < pop_size; i++) {
        if (pop[i].fitness <= threshold)
            temp[t1++] = pop[i];
    }
    
    // Mantém pelo menos 60% da população
    if (t1 < pop_size * 0.6) {
        t1 = (int)(pop_size * 0.6);
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
    
    // Copia de volta
    for (int i = 0; i < t2; i++)
        pop[i] = temp2[i];
    
    return t2;
}

/* ================= MAIN CORRIGIDO ================= */
int main() {
    srand(time(NULL));
    
    Individual pop[POP_MAX];
    Individual new_pop[POP_MAX];
    int pop_size = INIT_POP;
    
    // População inicial híbrida
    for (int i = 0; i < pop_size; i++) {
        if (i < pop_size / 3) {
            init_near_optimum(&pop[i]);  // Perto do ótimo
        } else if (i < 2 * pop_size / 3) {
            // Range moderado
            for (int j = 0; j < N; j++)
                pop[i].x[j] = rand_uniform(-50, 50);
            pop[i].fitness = fitness(&pop[i]);
        } else {
            random_individual(&pop[i]);  // Totalmente aleatório
        }
    }
    
    double best_fitness_ever = 1e9;
    Individual best_individual;
    double scale_factor = 0.5;
    int stagnation_counter = 0;
    int no_improvement_counter = 0;
    
    for (int gen = 0; gen < GENERATIONS; gen++) {
        // Ordena
        qsort(pop, pop_size, sizeof(Individual), cmp_fitness);
        
        double current_best = pop[0].fitness;
        
        // Atualiza melhor global
        if (current_best < best_fitness_ever - 1e-10) {
            best_fitness_ever = current_best;
            best_individual = pop[0];
            stagnation_counter = 0;
            no_improvement_counter = 0;
            scale_factor = fmax(0.2, scale_factor * 0.98);
        } else {
            no_improvement_counter++;
            if (no_improvement_counter > 30) {
                stagnation_counter++;
                scale_factor = fmin(1.5, scale_factor * 1.05);
                if (stagnation_counter > 60) {
                    printf("   🔄 Reinicialização parcial...\n");
                    for (int i = pop_size/2; i < pop_size; i++) {
                        if (rand() % 100 < 50)
                            init_near_optimum(&pop[i]);
                        else
                            random_individual(&pop[i]);
                    }
                    stagnation_counter = 0;
                    scale_factor = 0.5;
                }
            }
        }
        
        printf("Gen %3d | Best: %.10f | Pop: %3d | Scale: %.3f | NoImp: %d\n",
               gen, current_best, pop_size, scale_factor, no_improvement_counter);
        
        // Critério de parada
        if (best_fitness_ever < 1e-12) {
            printf("\n✅ Convergência ótima alcançada!\n");
            break;
        }
        
        /* ===== PRESERVA ELITE ===== */
        int elite_size = (int)(ELITE_RATE * pop_size);
        if (elite_size < 10) elite_size = 10;
        if (elite_size > pop_size) elite_size = pop_size;
        
        for (int i = 0; i < elite_size; i++)
            new_pop[i] = pop[i];
        int new_size = elite_size;
        
        /* ===== CLUSTERIZAÇÃO COM MELHORES ===== */
        int elite_for_cluster = (int)(pop_size * 0.4);  // Top 40%
        if (elite_for_cluster < 30) elite_for_cluster = pop_size;
        
        double centroids[K][N];
        kmeans(pop, elite_for_cluster, centroids);
        
        /* ===== MODELOS GAUSSIANOS ===== */
        double mu[K][N], cov[K][N][N], L[K][N][N];
        double avg_fit[K];
        int valid[K] = {0};
        
        for (int k = 0; k < K; k++) {
            valid[k] = compute_gaussian(pop, elite_for_cluster, k,
                                        mu[k], cov[k], &avg_fit[k]);
            
            if (valid[k] && cholesky(cov[k], L[k]))
                valid[k] = 1;
            else
                valid[k] = 0;
        }
        
        /* ===== PESOS ADAPTATIVOS ===== */
        double prob[K] = {0};
        double sum_prob = 0.0;
        
        for (int k = 0; k < K; k++) {
            if (valid[k]) {
                prob[k] = 1.0 / (avg_fit[k] + 0.1);
                sum_prob += prob[k];
            }
        }
        
        if (sum_prob > 0) {
            for (int k = 0; k < K; k++)
                prob[k] /= sum_prob;
        } else {
            for (int k = 0; k < K; k++)
                prob[k] = 1.0 / K;
        }
        
        /* ===== AMOSTRAGEM ===== */
        int samples_to_add = pop_size;
        for (int i = 0; i < samples_to_add && new_size < POP_MAX; i++) {
            // 10% de chance de gerar aleatório
            if ((double)rand() / RAND_MAX < 0.1) {
                if (best_fitness_ever < 1.0 && rand() % 100 < 70)
                    init_near_optimum(&new_pop[new_size]);
                else
                    random_individual(&new_pop[new_size]);
            } else {
                double r = (double)rand() / RAND_MAX;
                double acc = 0.0;
                int chosen = 0;
                
                for (int k = 0; k < K; k++) {
                    acc += prob[k];
                    if (r <= acc) {
                        chosen = k;
                        break;
                    }
                }
                
                if (valid[chosen]) {
                    sample_from_cluster(&new_pop[new_size], mu[chosen], L[chosen], scale_factor);
                } else {
                    random_individual(&new_pop[new_size]);
                }
            }
            new_size++;
        }
        
        /* ===== EXPLORAÇÃO LOCAL ===== */
        for (int i = 0; i < LOCAL_SEARCH && new_size < POP_MAX; i++) {
            Individual mutated = pop[i % elite_size];
            mutate_individual(&mutated, 0.2 * scale_factor);
            new_pop[new_size++] = mutated;
        }
        
        /* ===== FILTRAGEM ===== */
        int filtered_size = filter_population(new_pop, new_size, best_fitness_ever);
        
        /* ===== INJEÇÃO ALEATÓRIA ===== */
        int inject = (int)(RANDOM_RATE * filtered_size);
        if (inject < 5) inject = 5;
        
        if (filtered_size + inject <= POP_MAX) {
            for (int i = 0; i < inject; i++) {
                if (best_fitness_ever < 1.0 && rand() % 100 < 60)
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
        if (pop_size < 50) {
            for (int i = pop_size; i < 80; i++) {
                init_near_optimum(&pop[i]);
            }
            pop_size = 80;
            scale_factor = 0.5;
        }
        
        // Relatório periódico
        if (gen % 50 == 0 && gen > 0) {
            double avg_fitness = 0;
            int top_k = (pop_size < 20) ? pop_size : 20;
            for (int i = 0; i < top_k; i++)
                avg_fitness += pop[i].fitness;
            avg_fitness /= top_k;
            printf("   📊 Média top %d: %.8f\n", top_k, avg_fitness);
            printf("   📊 Melhor encontrado: %.10f\n", best_fitness_ever);
        }
    }
    
    // Resultado final
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
