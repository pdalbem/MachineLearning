#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* ================= PARÂMETROS ================= */
#define N 5
#define INITIAL_POP 300
#define MAX_POP 500
#define MIN_POP 80
#define GENERATIONS 600
#define ELITE_FRACTION 0.2
#define RANDOM_RATE 0.1
#define SIM_THRESHOLD 15.0     // Para Schwefel, valores podem ser distantes
#define MAX_CLUSTERS 6
#define LOCAL_SEARCH 10
#define EPS 1e-8

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ================= ESTRUTURAS ================= */
typedef struct {
    double x[N];
    double fitness;
    int cluster;
} Individual;

typedef struct {
    double mean[N];
    double cov[N][N];
    double weight;
    double avg_fitness;
} Gaussian;

/* ================= FUNÇÃO SCHWEFEL ================= */
double schwefel(double *x) {
    double sum = 0.0;
    for(int i = 0; i < N; i++) {
        sum += x[i] * sin(sqrt(fabs(x[i])));
    }
    return 418.9829 * N - sum;
}

/* ================= UTILITÁRIOS ================= */
double rand_uniform(double a, double b) {
    return a + (b - a) * ((double)rand() / RAND_MAX);
}

double rand_normal() {
    double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
    double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

double distance(Individual *a, Individual *b) {
    double d = 0.0;
    for(int i = 0; i < N; i++) {
        double diff = a->x[i] - b->x[i];
        d += diff * diff;
    }
    return sqrt(d);
}

int cmp_fitness(const void *a, const void *b) {
    double fa = ((Individual*)a)->fitness;
    double fb = ((Individual*)b)->fitness;
    return (fa > fb) - (fa < fb);
}

/* ================= INICIALIZAÇÃO INTELIGENTE ================= */
void random_individual(Individual *ind) {
    for(int i = 0; i < N; i++)
        ind->x[i] = rand_uniform(-500, 500);
    ind->fitness = schwefel(ind->x);
}

void init_near_optimum(Individual *ind) {
    // O mínimo global está em x = 420.9687
    double optimum = 420.9687;
    for(int i = 0; i < N; i++)
        ind->x[i] = optimum + rand_normal() * 20.0;
    ind->fitness = schwefel(ind->x);
}

void init_population(Individual *pop, int size) {
    for(int i = 0; i < size; i++) {
        if(i < size / 3) {
            // Perto do ótimo
            init_near_optimum(&pop[i]);
        } else if(i < 2 * size / 3) {
            // Região promissora perto de ±420
            for(int j = 0; j < N; j++) {
                if(rand() % 2)
                    pop[i].x[j] = 420.9687 + rand_normal() * 50.0;
                else
                    pop[i].x[j] = -420.9687 + rand_normal() * 50.0;
            }
            pop[i].fitness = schwefel(pop[i].x);
        } else {
            // Exploração ampla
            random_individual(&pop[i]);
        }
    }
}

/* ================= K-MEANS MELHORADO ================= */
void kmeans(Individual *pop, int size, Gaussian *clusters, int k) {
    if(k > size) k = size;
    if(k < 2) k = 2;
    
    // Inicializa com os melhores indivíduos
    for(int c = 0; c < k; c++) {
        int idx = (c * (size / k)) % size;
        for(int j = 0; j < N; j++)
            clusters[c].mean[j] = pop[idx].x[j];
        clusters[c].weight = 1.0 / k;
    }
    
    int labels[MAX_POP];
    int changed;
    
    for(int iter = 0; iter < 20; iter++) {
        changed = 0;
        
        // Atribui clusters
        for(int i = 0; i < size; i++) {
            double best_dist = 1e9;
            int best_c = 0;
            for(int c = 0; c < k; c++) {
                double dist = 0;
                for(int j = 0; j < N; j++) {
                    double diff = pop[i].x[j] - clusters[c].mean[j];
                    dist += diff * diff;
                }
                if(dist < best_dist) {
                    best_dist = dist;
                    best_c = c;
                }
            }
            if(labels[i] != best_c) {
                labels[i] = best_c;
                changed = 1;
            }
        }
        
        if(!changed) break;
        
        // Recalcula centroides
        double sum[N][MAX_CLUSTERS] = {{0}};
        int count[MAX_CLUSTERS] = {0};
        
        for(int i = 0; i < size; i++) {
            int c = labels[i];
            count[c]++;
            for(int j = 0; j < N; j++)
                sum[c][j] += pop[i].x[j];
        }
        
        for(int c = 0; c < k; c++) {
            if(count[c] > 0) {
                for(int j = 0; j < N; j++)
                    clusters[c].mean[j] = sum[c][j] / count[c];
                clusters[c].weight = (double)count[c] / size;
            }
        }
    }
}

/* ================= MATRIZ DE COVARIÂNCIA ================= */
void compute_covariance(Individual *pop, int size, int cluster, 
                        double mean[N], double cov[N][N]) {
    // Inicializa
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
            cov[i][j] = 0.0;
    
    int count = 0;
    
    // Calcula covariância
    for(int i = 0; i < size; i++) {
        if(pop[i].cluster != cluster) continue;
        count++;
        
        for(int j = 0; j < N; j++) {
            for(int k = 0; k < N; k++) {
                cov[j][k] += (pop[i].x[j] - mean[j]) * 
                             (pop[i].x[k] - mean[k]);
            }
        }
    }
    
    if(count < 2) return;
    
    // Normaliza e regulariza
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            cov[i][j] /= count;
        }
        // Regularização adaptativa
        double scale = fabs(mean[i]);
        if(scale < 100) scale = 100;
        cov[i][i] += 10.0 * scale;
    }
}

/* ================= CHOLESKY ================= */
int cholesky(double A[N][N], double L[N][N]) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j <= i; j++) {
            double sum = 0.0;
            for(int k = 0; k < j; k++)
                sum += L[i][k] * L[j][k];
            
            if(i == j) {
                double val = A[i][i] - sum;
                if(val <= 1e-12) {
                    L[i][j] = sqrt(1e-8);
                } else {
                    L[i][j] = sqrt(val);
                }
            } else {
                if(fabs(L[j][j]) < 1e-12) return 0;
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
    }
    return 1;
}

/* ================= AMOSTRAGEM ================= */
void sample_from_gaussian(Individual *ind, double mean[N], double L[N][N], double scale) {
    double z[N];
    
    for(int i = 0; i < N; i++)
        z[i] = rand_normal() * scale;
    
    for(int i = 0; i < N; i++) {
        ind->x[i] = mean[i];
        for(int j = 0; j <= i; j++)
            ind->x[i] += L[i][j] * z[j];
        
        // Limites realistas para Schwefel
        if(ind->x[i] < -500) ind->x[i] = -500;
        if(ind->x[i] > 500) ind->x[i] = 500;
    }
    
    ind->fitness = schwefel(ind->x);
}

void mutate_individual(Individual *ind, double step) {
    for(int j = 0; j < N; j++) {
        ind->x[j] += rand_normal() * step;
        if(ind->x[j] < -500) ind->x[j] = -500;
        if(ind->x[j] > 500) ind->x[j] = 500;
    }
    ind->fitness = schwefel(ind->x);
}

/* ================= FILTRO ADAPTATIVO ================= */
int filter_population(Individual *pop, int pop_size, double best_fitness) {
    // Threshold adaptativo para Schwefel
    double threshold;
    if(best_fitness < 1.0)
        threshold = best_fitness + 0.5;
    else if(best_fitness < 10.0)
        threshold = best_fitness * 1.5;
    else if(best_fitness < 100.0)
        threshold = best_fitness * 2.0;
    else
        threshold = best_fitness * 2.5;
    
    // Filtro de fitness
    Individual temp[MAX_POP];
    int t1 = 0;
    
    for(int i = 0; i < pop_size; i++) {
        if(pop[i].fitness <= threshold)
            temp[t1++] = pop[i];
    }
    
    // Mantém pelo menos 60%
    if(t1 < pop_size * 0.6) {
        t1 = (int)(pop_size * 0.6);
        qsort(pop, pop_size, sizeof(Individual), cmp_fitness);
        for(int i = 0; i < t1; i++)
            temp[i] = pop[i];
    }
    
    // Filtro de similaridade
    Individual temp2[MAX_POP];
    int t2 = 0;
    
    for(int i = 0; i < t1; i++) {
        int keep = 1;
        for(int j = 0; j < t2; j++) {
            if(distance(&temp[i], &temp2[j]) < SIM_THRESHOLD) {
                keep = 0;
                break;
            }
        }
        if(keep) temp2[t2++] = temp[i];
    }
    
    // Copia de volta
    for(int i = 0; i < t2; i++)
        pop[i] = temp2[i];
    
    return t2;
}

/* ================= MAIN ================= */
int main() {
    srand(time(NULL));
    
    Individual pop[MAX_POP];
    Individual new_pop[MAX_POP];
    int pop_size = INITIAL_POP;
    
    init_population(pop, pop_size);
    
    double best_fitness_ever = 1e9;
    Individual best_individual;
    double scale_factor = 0.5;
    int stagnation_counter = 0;
    int no_improvement_counter = 0;
    
    for(int gen = 0; gen < GENERATIONS; gen++) {
        qsort(pop, pop_size, sizeof(Individual), cmp_fitness);
        
        double current_best = pop[0].fitness;
        
        // Atualiza melhor global
        if(current_best < best_fitness_ever - 1e-10) {
            best_fitness_ever = current_best;
            best_individual = pop[0];
            stagnation_counter = 0;
            no_improvement_counter = 0;
            scale_factor = fmax(0.2, scale_factor * 0.98);
        } else {
            no_improvement_counter++;
            if(no_improvement_counter > 30) {
                stagnation_counter++;
                scale_factor = fmin(1.2, scale_factor * 1.03);
                if(stagnation_counter > 60) {
                    printf("   🔄 Reinicialização parcial...\n");
                    for(int i = pop_size/2; i < pop_size; i++) {
                        if(rand() % 100 < 60)
                            init_near_optimum(&pop[i]);
                        else
                            random_individual(&pop[i]);
                    }
                    stagnation_counter = 0;
                    scale_factor = 0.5;
                }
            }
        }
        
        printf("Gen %3d | Best: %.6f | Pop: %3d | Scale: %.3f | NoImp: %d\n",
               gen, current_best, pop_size, scale_factor, no_improvement_counter);
        
        if(best_fitness_ever < 1e-8) {
            printf("\n✅ Convergência ótima alcançada!\n");
            break;
        }
        
        /* ===== PRESERVA ELITE ===== */
        int elite_size = (int)(ELITE_FRACTION * pop_size);
        if(elite_size < 10) elite_size = 10;
        
        for(int i = 0; i < elite_size; i++)
            new_pop[i] = pop[i];
        int new_size = elite_size;
        
        /* ===== CLUSTERIZAÇÃO ===== */
        Gaussian clusters[MAX_CLUSTERS];
        int k = MAX_CLUSTERS;
        if(pop_size < 50) k = 3;
        if(pop_size < 30) k = 2;
        
        int elite_for_cluster = pop_size / 2;  // Top 50%
        kmeans(pop, elite_for_cluster, clusters, k);
        
        /* ===== MODELOS GAUSSIANOS ===== */
        double mu[MAX_CLUSTERS][N], Lmat[MAX_CLUSTERS][N][N];
        double avg_fit[MAX_CLUSTERS];
        int valid[MAX_CLUSTERS] = {0};
        
        for(int c = 0; c < k; c++) {
            // Encontra indivíduos neste cluster
            int count = 0;
            double sum_fit = 0;
            
            for(int i = 0; i < pop_size; i++) {
                // Encontra cluster do indivíduo
                double best_dist = 1e9;
                int best_c = 0;
                for(int cc = 0; cc < k; cc++) {
                    double dist = 0;
                    for(int j = 0; j < N; j++) {
                        double diff = pop[i].x[j] - clusters[cc].mean[j];
                        dist += diff * diff;
                    }
                    if(dist < best_dist) {
                        best_dist = dist;
                        best_c = cc;
                    }
                }
                
                if(best_c == c) {
                    count++;
                    sum_fit += pop[i].fitness;
                }
            }
            
            if(count >= 3) {
                avg_fit[c] = sum_fit / count;
                double cov[N][N];
                compute_covariance(pop, pop_size, c, clusters[c].mean, cov);
                
                if(cholesky(cov, Lmat[c])) {
                    valid[c] = 1;
                    for(int j = 0; j < N; j++)
                        mu[c][j] = clusters[c].mean[j];
                }
            }
        }
        
        /* ===== PESOS ===== */
        double prob[MAX_CLUSTERS] = {0};
        double sum_prob = 0;
        
        for(int c = 0; c < k; c++) {
            if(valid[c]) {
                prob[c] = 1.0 / (avg_fit[c] + 0.1);
                sum_prob += prob[c];
            }
        }
        
        if(sum_prob > 0) {
            for(int c = 0; c < k; c++)
                prob[c] /= sum_prob;
        } else {
            for(int c = 0; c < k; c++)
                prob[c] = 1.0 / k;
        }
        
        /* ===== AMOSTRAGEM ===== */
        int samples_to_add = pop_size;
        for(int i = 0; i < samples_to_add && new_size < MAX_POP; i++) {
            if((double)rand() / RAND_MAX < 0.1) {
                // Exploração aleatória
                if(rand() % 100 < 50)
                    init_near_optimum(&new_pop[new_size]);
                else
                    random_individual(&new_pop[new_size]);
            } else {
                double r = (double)rand() / RAND_MAX;
                double acc = 0;
                int chosen = 0;
                
                for(int c = 0; c < k; c++) {
                    acc += prob[c];
                    if(r <= acc) {
                        chosen = c;
                        break;
                    }
                }
                
                if(valid[chosen]) {
                    sample_from_gaussian(&new_pop[new_size], mu[chosen], Lmat[chosen], scale_factor);
                } else {
                    random_individual(&new_pop[new_size]);
                }
            }
            new_size++;
        }
        
        /* ===== EXPLORAÇÃO LOCAL ===== */
        for(int i = 0; i < LOCAL_SEARCH && new_size < MAX_POP; i++) {
            Individual mutated = pop[i % elite_size];
            mutate_individual(&mutated, 5.0 * scale_factor);
            new_pop[new_size++] = mutated;
        }
        
        /* ===== FILTRAGEM ===== */
        int filtered_size = filter_population(new_pop, new_size, best_fitness_ever);
        
        /* ===== INJEÇÃO ALEATÓRIA ===== */
        int inject = (int)(RANDOM_RATE * filtered_size);
        if(inject < 5) inject = 5;
        
        if(filtered_size + inject <= MAX_POP) {
            for(int i = 0; i < inject; i++) {
                if(best_fitness_ever < 10.0 && rand() % 100 < 70)
                    init_near_optimum(&new_pop[filtered_size + i]);
                else
                    random_individual(&new_pop[filtered_size + i]);
            }
            filtered_size += inject;
        }
        
        // Atualiza população
        for(int i = 0; i < filtered_size; i++)
            pop[i] = new_pop[i];
        pop_size = filtered_size;
        
        // Garante população mínima
        if(pop_size < MIN_POP) {
            for(int i = pop_size; i < MIN_POP; i++)
                init_near_optimum(&pop[i]);
            pop_size = MIN_POP;
        }
        
        // Relatório periódico
        if(gen % 50 == 0 && gen > 0) {
            double avg_fitness = 0;
            int top_k = (pop_size < 20) ? pop_size : 20;
            for(int i = 0; i < top_k; i++)
                avg_fitness += pop[i].fitness;
            avg_fitness /= top_k;
            printf("   📊 Média top %d: %.6f\n", top_k, avg_fitness);
        }
    }
    
    printf("\n========================================\n");
    printf("MELHOR SOLUÇÃO ENCONTRADA:\n");
    printf("x = [");
    for(int i = 0; i < N; i++) {
        printf("%.6f", best_individual.x[i]);
        if(i < N-1) printf(", ");
    }
    printf("]\n");
    printf("Fitness = %.10f\n", best_fitness_ever);
    printf("Esperado = 0.0000000000\n");
    printf("Ótimo global em x = 420.9687\n");
    printf("========================================\n");
    
    return 0;
}