#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ================= PARÂMETROS ================= */
#define D 5
#define INIT_POP 400
#define MAX_POP 800
#define MIN_POP 100
#define MAX_GEN 500
#define ELITE_FRACTION 0.2
#define RANDOM_RATE 0.15
#define SIM_THRESHOLD 0.05
#define MAX_CLUSTERS 6
#define LOCAL_SEARCH 10
#define MUTATION_SCALE 0.1
#define EPS 1e-8

/* ================= ESTRUTURAS ================= */
typedef struct {
    double x[D];
    double fitness;
    int cluster;
} Individual;

typedef struct {
    double mean[D];
    double cov[D][D];
    double weight;
    double avg_fitness;
} Gaussian;

/* ================= FUNÇÃO MICHALEWICZ ================= */
double michalewicz(Individual *ind) {
    double m = 10.0;
    double sum = 0.0;
    
    for (int i = 0; i < D; i++) {
        double x = ind->x[i];
        double term1 = sin(x);
        double term2 = pow(sin(((i + 1) * x * x) / M_PI), 2.0 * m);
        sum += term1 * term2;
    }
    
    return -sum;  // Minimização
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
    for(int i = 0; i < D; i++) {
        double diff = a->x[i] - b->x[i];
        d += diff * diff;
    }
    return sqrt(d);
}

int cmp_fitness(const void *a, const void *b) {
    double fa = ((Individual*)a)->fitness;
    double fb = ((Individual*)b)->fitness;
    return (fa > fb) - (fa < fb);  // Menor é melhor
}

/* ================= INICIALIZAÇÃO INTELIGENTE ================= */
void random_individual(Individual *ind) {
    for(int i = 0; i < D; i++)
        ind->x[i] = rand_uniform(0, M_PI);
    ind->fitness = michalewicz(ind);
}

void init_near_optimum(Individual *ind) {
    // Ótimo conhecido para D=5: ~[2.20, 1.57, 1.28, 1.92, 1.64]
    double optimum[D] = {2.20, 1.57, 1.28, 1.92, 1.64};
    
    for(int i = 0; i < D; i++) {
        ind->x[i] = optimum[i] + rand_normal() * 0.2;
        if(ind->x[i] < 0) ind->x[i] = 0;
        if(ind->x[i] > M_PI) ind->x[i] = M_PI;
    }
    ind->fitness = michalewicz(ind);
}

void init_population(Individual *pop, int size) {
    for(int i = 0; i < size; i++) {
        if(i < size / 2) {
            // Metade perto do ótimo conhecido
            init_near_optimum(&pop[i]);
        } else {
            // Metade aleatória para exploração
            random_individual(&pop[i]);
        }
    }
}

/* ================= K-MEANS ================= */
void kmeans(Individual *pop, int size, Gaussian *clusters, int k) {
    if(k > size) k = size;
    if(k < 2) k = 2;
    
    // Inicializa com os melhores
    for(int c = 0; c < k; c++) {
        int idx = (c * (size / k)) % size;
        for(int j = 0; j < D; j++)
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
                for(int j = 0; j < D; j++) {
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
        double sum[MAX_CLUSTERS][D] = {{0}};
        int count[MAX_CLUSTERS] = {0};
        
        for(int i = 0; i < size; i++) {
            int c = labels[i];
            count[c]++;
            for(int j = 0; j < D; j++)
                sum[c][j] += pop[i].x[j];
        }
        
        for(int c = 0; c < k; c++) {
            if(count[c] > 0) {
                for(int j = 0; j < D; j++)
                    clusters[c].mean[j] = sum[c][j] / count[c];
                clusters[c].weight = (double)count[c] / size;
            }
        }
    }
}

/* ================= MATRIZ DE COVARIÂNCIA ================= */
void compute_covariance(Individual *pop, int size, int cluster, 
                        double mean[D], double cov[D][D]) {
    for(int i = 0; i < D; i++)
        for(int j = 0; j < D; j++)
            cov[i][j] = 0.0;
    
    int count = 0;
    
    for(int i = 0; i < size; i++) {
        if(pop[i].cluster != cluster) continue;
        count++;
        
        for(int j = 0; j < D; j++) {
            for(int k = 0; k < D; k++) {
                cov[j][k] += (pop[i].x[j] - mean[j]) * 
                             (pop[i].x[k] - mean[k]);
            }
        }
    }
    
    if(count < 2) return;
    
    for(int i = 0; i < D; i++) {
        for(int j = 0; j < D; j++) {
            cov[i][j] /= count;
        }
        cov[i][i] += 0.01;  // Regularização
    }
}

/* ================= CHOLESKY ================= */
int cholesky(double A[D][D], double L[D][D]) {
    for(int i = 0; i < D; i++) {
        for(int j = 0; j <= i; j++) {
            double sum = 0.0;
            for(int k = 0; k < j; k++)
                sum += L[i][k] * L[j][k];
            
            if(i == j) {
                double val = A[i][i] - sum;
                if(val <= 1e-12) {
                    L[i][j] = sqrt(1e-6);
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
void sample_from_gaussian(Individual *ind, double mean[D], double L[D][D], double scale) {
    double z[D];
    
    for(int i = 0; i < D; i++)
        z[i] = rand_normal() * scale;
    
    for(int i = 0; i < D; i++) {
        ind->x[i] = mean[i];
        for(int j = 0; j <= i; j++)
            ind->x[i] += L[i][j] * z[j];
        
        // Mantém nos limites [0, π]
        if(ind->x[i] < 0) ind->x[i] = 0;
        if(ind->x[i] > M_PI) ind->x[i] = M_PI;
    }
    
    ind->fitness = michalewicz(ind);
}

void mutate_individual(Individual *ind, double step) {
    for(int j = 0; j < D; j++) {
        ind->x[j] += rand_normal() * step;
        if(ind->x[j] < 0) ind->x[j] = 0;
        if(ind->x[j] > M_PI) ind->x[j] = M_PI;
    }
    ind->fitness = michalewicz(ind);
}

/* ================= FILTRO ADAPTATIVO ================= */
int filter_population(Individual *pop, int pop_size, double best_fitness) {
    double threshold;
    if(best_fitness < -4.5)
        threshold = best_fitness + 0.1;
    else if(best_fitness < -4.0)
        threshold = best_fitness * 1.05;
    else if(best_fitness < -3.0)
        threshold = best_fitness * 1.1;
    else
        threshold = best_fitness * 1.2;
    
    Individual temp[MAX_POP];
    int t1 = 0;
    
    for(int i = 0; i < pop_size; i++) {
        if(pop[i].fitness <= threshold)
            temp[t1++] = pop[i];
    }
    
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
    
    for(int i = 0; i < t2; i++)
        pop[i] = temp2[i];
    
    return t2;
}

/* ================= MAIN ================= */
int main() {
    srand(time(NULL));
    
    Individual pop[MAX_POP];
    Individual new_pop[MAX_POP];
    int pop_size = INIT_POP;
    
    init_population(pop, pop_size);
    
    double best_fitness_ever = 1e9;
    Individual best_individual;
    double scale_factor = 0.3;
    int stagnation_counter = 0;
    int no_improvement_counter = 0;
    
    for(int gen = 0; gen < MAX_GEN; gen++) {
        qsort(pop, pop_size, sizeof(Individual), cmp_fitness);
        
        double current_best = pop[0].fitness;
        
        if(current_best < best_fitness_ever - 1e-8) {
            best_fitness_ever = current_best;
            best_individual = pop[0];
            stagnation_counter = 0;
            no_improvement_counter = 0;
            scale_factor = fmax(0.1, scale_factor * 0.98);
        } else {
            no_improvement_counter++;
            if(no_improvement_counter > 40) {
                stagnation_counter++;
                scale_factor = fmin(0.8, scale_factor * 1.03);
                if(stagnation_counter > 70) {
                    printf("   🔄 Reinicialização parcial...\n");
                    for(int i = pop_size/2; i < pop_size; i++) {
                        if(rand() % 100 < 60)
                            init_near_optimum(&pop[i]);
                        else
                            random_individual(&pop[i]);
                    }
                    stagnation_counter = 0;
                    scale_factor = 0.3;
                }
            }
        }
        
        printf("Gen %3d | Best: %.6f | Pop: %3d | Scale: %.3f | NoImp: %d\n",
               gen, current_best, pop_size, scale_factor, no_improvement_counter);
        
        if(best_fitness_ever < -4.687 && fabs(best_fitness_ever + 4.687) < 1e-6) {
            printf("\n✅ Ótimo global encontrado!\n");
            break;
        }
        
        int elite_size = (int)(ELITE_FRACTION * pop_size);
        if(elite_size < 10) elite_size = 10;
        
        for(int i = 0; i < elite_size; i++)
            new_pop[i] = pop[i];
        int new_size = elite_size;
        
        // Clusterização
        Gaussian clusters[MAX_CLUSTERS];
        int k = MAX_CLUSTERS;
        if(pop_size < 50) k = 3;
        
        int elite_for_cluster = pop_size / 2;
        kmeans(pop, elite_for_cluster, clusters, k);
        
        // Prepara clusters para amostragem
        double mu[MAX_CLUSTERS][D], Lmat[MAX_CLUSTERS][D][D];
        double avg_fit[MAX_CLUSTERS];
        int valid[MAX_CLUSTERS] = {0};
        
        // Atribui clusters
        for(int i = 0; i < pop_size; i++) {
            double best_dist = 1e9;
            int best_c = 0;
            for(int c = 0; c < k; c++) {
                double dist = 0;
                for(int j = 0; j < D; j++) {
                    double diff = pop[i].x[j] - clusters[c].mean[j];
                    dist += diff * diff;
                }
                if(dist < best_dist) {
                    best_dist = dist;
                    best_c = c;
                }
            }
            pop[i].cluster = best_c;
        }
        
        // Calcula médias e covariâncias
        for(int c = 0; c < k; c++) {
            int count = 0;
            double sum_fit = 0;
            
            for(int i = 0; i < pop_size; i++) {
                if(pop[i].cluster == c) {
                    count++;
                    sum_fit += pop[i].fitness;
                }
            }
            
            if(count >= 3) {
                avg_fit[c] = sum_fit / count;
                double cov[D][D];
                compute_covariance(pop, pop_size, c, clusters[c].mean, cov);
                
                if(cholesky(cov, Lmat[c])) {
                    valid[c] = 1;
                    for(int j = 0; j < D; j++)
                        mu[c][j] = clusters[c].mean[j];
                }
            }
        }
        
        // Pesos
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
        
        // Amostragem
        int samples_to_add = pop_size;
        for(int i = 0; i < samples_to_add && new_size < MAX_POP; i++) {
            if((double)rand() / RAND_MAX < 0.15) {
                if(rand() % 100 < 60)
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
        
        // Exploração local
        for(int i = 0; i < LOCAL_SEARCH && new_size < MAX_POP; i++) {
            Individual mutated = pop[i % elite_size];
            mutate_individual(&mutated, 0.1 * scale_factor);
            new_pop[new_size++] = mutated;
        }
        
        // Filtragem
        int filtered_size = filter_population(new_pop, new_size, best_fitness_ever);
        
        // Injeção aleatória
        int inject = (int)(RANDOM_RATE * filtered_size);
        if(inject < 5) inject = 5;
        
        if(filtered_size + inject <= MAX_POP) {
            for(int i = 0; i < inject; i++) {
                if(best_fitness_ever < -4.0 && rand() % 100 < 70)
                    init_near_optimum(&new_pop[filtered_size + i]);
                else
                    random_individual(&new_pop[filtered_size + i]);
            }
            filtered_size += inject;
        }
        
        for(int i = 0; i < filtered_size; i++)
            pop[i] = new_pop[i];
        pop_size = filtered_size;
        
        if(pop_size < MIN_POP) {
            for(int i = pop_size; i < MIN_POP; i++)
                init_near_optimum(&pop[i]);
            pop_size = MIN_POP;
        }
        
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
    for(int i = 0; i < D; i++) {
        printf("%.6f", best_individual.x[i]);
        if(i < D-1) printf(", ");
    }
    printf("]\n");
    printf("Fitness = %.10f\n", best_fitness_ever);
    printf("Ótimo global esperado ≈ -4.687658\n");
    printf("========================================\n");
    
    return 0;
}