#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ================= PARÂMETROS ================= */
#define DIM 5
#define INITIAL_POP 200
#define MAX_POP 300
#define MIN_POP 50
#define MAX_GEN 300
#define RANDOM_RATE 0.15
#define SIM_THRESHOLD 0.05
#define MAX_CLUSTERS 8
#define ELITE_RATE 0.20    // 20% de elite
#define LOCAL_SEARCH 5

typedef struct {
    double x[DIM];
    double fitness;
} Individual;

/* ================= FUNÇÃO ACKLEY ================= */
double ackley(double *x) {
    double a = 20, b = 0.2, c = 2*M_PI;
    double sum1 = 0, sum2 = 0;
    
    for(int i = 0; i < DIM; i++) {
        sum1 += x[i] * x[i];
        sum2 += cos(c * x[i]);
    }
    
    return -a * exp(-b * sqrt(sum1/DIM)) - exp(sum2/DIM) + a + exp(1);
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
    double d = 0;
    for(int i = 0; i < DIM; i++) {
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

/* ================= CLUSTER ================= */
typedef struct {
    double mean[DIM];
    double sigma[DIM];
    int count;
    double avg_fitness;
} Cluster;

Cluster clusters[MAX_CLUSTERS];
int num_clusters;

void cluster_population(Individual pop[], int pop_size) {
    num_clusters = MAX_CLUSTERS;
    if(pop_size < num_clusters) num_clusters = pop_size;
    if(num_clusters < 2) num_clusters = 2;
    
    // Inicializa centroides com os MELHORES indivíduos
    for(int k = 0; k < num_clusters; k++) {
        int idx = k % (pop_size / 2);  // Pega dos 50% melhores
        for(int d = 0; d < DIM; d++)
            clusters[k].mean[d] = pop[idx].x[d];
        clusters[k].count = 0;
        clusters[k].avg_fitness = 1e9;
    }
    
    // K-means com mais iterações
    for(int iter = 0; iter < 20; iter++) {
        double sum_x[MAX_CLUSTERS][DIM] = {{0}};
        int count[MAX_CLUSTERS] = {0};
        
        // Atribui indivíduos ao cluster mais próximo
        for(int i = 0; i < pop_size; i++) {
            double best_dist = 1e9;
            int best_k = 0;
            
            for(int k = 0; k < num_clusters; k++) {
                double dist = 0;
                for(int j = 0; j < DIM; j++) {
                    double diff = pop[i].x[j] - clusters[k].mean[j];
                    dist += diff * diff;
                }
                if(dist < best_dist) {
                    best_dist = dist;
                    best_k = k;
                }
            }
            
            for(int j = 0; j < DIM; j++)
                sum_x[best_k][j] += pop[i].x[j];
            count[best_k]++;
        }
        
        // Atualiza centroides
        int changed = 0;
        for(int k = 0; k < num_clusters; k++) {
            if(count[k] == 0) continue;
            
            for(int j = 0; j < DIM; j++) {
                double new_mean = sum_x[k][j] / count[k];
                if(fabs(new_mean - clusters[k].mean[j]) > 1e-8)
                    changed = 1;
                clusters[k].mean[j] = new_mean;
            }
            clusters[k].count = count[k];
        }
        
        if(!changed) break;
    }
    
    // Calcula sigma e fitness médio por cluster
    for(int k = 0; k < num_clusters; k++) {
        if(clusters[k].count < 2) {
            // Cluster pequeno: usa sigma global
            for(int j = 0; j < DIM; j++)
                clusters[k].sigma[j] = 0.5;
            clusters[k].avg_fitness = 1e9;
            continue;
        }
        
        double var[DIM] = {0};
        double sum_fitness = 0;
        int count_in_cluster = 0;
        
        for(int i = 0; i < pop_size; i++) {
            // Verifica qual cluster este indivíduo pertence
            double best_dist = 1e9;
            int best_k = 0;
            for(int c = 0; c < num_clusters; c++) {
                double dist = 0;
                for(int j = 0; j < DIM; j++) {
                    double diff = pop[i].x[j] - clusters[c].mean[j];
                    dist += diff * diff;
                }
                if(dist < best_dist) {
                    best_dist = dist;
                    best_k = c;
                }
            }
            
            if(best_k == k) {
                for(int j = 0; j < DIM; j++) {
                    double diff = pop[i].x[j] - clusters[k].mean[j];
                    var[j] += diff * diff;
                }
                sum_fitness += pop[i].fitness;
                count_in_cluster++;
            }
        }
        
        // Normaliza variância
        for(int j = 0; j < DIM; j++) {
            var[j] /= count_in_cluster;
            clusters[k].sigma[j] = sqrt(var[j] + 0.01);  // Adiciona regularização
        }
        clusters[k].avg_fitness = sum_fitness / count_in_cluster;
    }
}

/* ================= AMOSTRAGEM ================= */
void sample_from_cluster(Individual *ind, int cluster_idx, double scale) {
    for(int j = 0; j < DIM; j++) {
        double z = rand_normal() * scale;
        ind->x[j] = clusters[cluster_idx].mean[j] + z * clusters[cluster_idx].sigma[j];
        
        // Mantém nos limites [-5, 5]
        if(ind->x[j] < -5) ind->x[j] = -5;
        if(ind->x[j] > 5) ind->x[j] = 5;
    }
    ind->fitness = ackley(ind->x);
}

void random_individual(Individual *ind) {
    for(int j = 0; j < DIM; j++)
        ind->x[j] = rand_uniform(-5, 5);
    ind->fitness = ackley(ind->x);
}

void mutate_individual(Individual *ind, double step) {
    for(int j = 0; j < DIM; j++) {
        ind->x[j] += rand_normal() * step;
        if(ind->x[j] < -5) ind->x[j] = -5;
        if(ind->x[j] > 5) ind->x[j] = 5;
    }
    ind->fitness = ackley(ind->x);
}

/* ================= FILTROS ================= */
int eliminate_weak_and_similar(Individual pop[], int pop_size, double best_fitness) {
    // Threshold adaptativo
    double threshold;
    if(best_fitness < 0.1)
        threshold = best_fitness + 0.05;
    else if(best_fitness < 1.0)
        threshold = best_fitness * 1.2;
    else
        threshold = best_fitness * 2.0;
    
    // Primeiro filtro: fitness
    Individual temp[MAX_POP];
    int t1 = 0;
    
    for(int i = 0; i < pop_size; i++) {
        if(pop[i].fitness <= threshold)
            temp[t1++] = pop[i];
    }
    
    // Segundo filtro: similaridade
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

/* ================= MAIN MELHORADO ================= */
int main() {
    srand(time(NULL));
    
    Individual pop[MAX_POP];
    Individual new_pop[MAX_POP];
    int pop_size = INITIAL_POP;
    
    // Inicialização híbrida
    for(int i = 0; i < pop_size; i++) {
        if(i < pop_size / 3) {
            // Próximo do ótimo (0,0,...,0)
            for(int j = 0; j < DIM; j++)
                pop[i].x[j] = rand_normal() * 0.5;
            pop[i].fitness = ackley(pop[i].x);
        } else {
            random_individual(&pop[i]);
        }
    }
    
    double best_fitness_ever = 1e9;
    Individual best_individual;
    double scale_factor = 1.0;
    int stagnation_counter = 0;
    
    for(int gen = 0; gen < MAX_GEN; gen++) {
        // Ordena população
        qsort(pop, pop_size, sizeof(Individual), cmp_fitness);
        
        double current_best = pop[0].fitness;
        
        // Atualiza melhor global
        if(current_best < best_fitness_ever) {
            best_fitness_ever = current_best;
            best_individual = pop[0];
            stagnation_counter = 0;
            // Reduz escala quando melhora
            scale_factor = fmax(0.3, scale_factor * 0.98);
        } else {
            stagnation_counter++;
            // Aumenta escala se estagnado
            if(stagnation_counter > 30) {
                scale_factor = fmin(2.0, scale_factor * 1.05);
                if(stagnation_counter > 50) {
                    // Reinicialização parcial
                    for(int i = pop_size/2; i < pop_size; i++)
                        random_individual(&pop[i]);
                    stagnation_counter = 0;
                }
            }
        }
        
        printf("Gen %3d | Best: %.10f | Pop: %3d | Scale: %.3f | Stag: %d\n",
               gen, current_best, pop_size, scale_factor, stagnation_counter);
        
        // Critério de parada
        if(best_fitness_ever < 1e-10) {
            printf("\n✅ Convergência alcançada!\n");
            break;
        }
        
        /* ===== PRESERVA ELITE ===== */
        int elite_size = (int)(ELITE_RATE * pop_size);
        if(elite_size < 5) elite_size = 5;
        
        for(int i = 0; i < elite_size; i++)
            new_pop[i] = pop[i];
        int new_size = elite_size;
        
        /* ===== CLUSTERIZAÇÃO ===== */
        cluster_population(pop, pop_size);
        
        /* ===== AMOSTRAGEM POR CLUSTER COM PESOS ===== */
        // Calcula pesos baseados na qualidade do cluster (menor fitness = maior peso)
        double weights[MAX_CLUSTERS] = {0};
        double sum_weights = 0;
        
        for(int k = 0; k < num_clusters; k++) {
            if(clusters[k].count > 0) {
                weights[k] = 1.0 / (clusters[k].avg_fitness + 0.1);
                sum_weights += weights[k];
            }
        }
        
        if(sum_weights > 0) {
            for(int k = 0; k < num_clusters; k++)
                weights[k] /= sum_weights;
        } else {
            for(int k = 0; k < num_clusters; k++)
                weights[k] = 1.0 / num_clusters;
        }
        
        // Gera novos indivíduos
        int samples_to_add = pop_size;
        for(int i = 0; i < samples_to_add && new_size < MAX_POP; i++) {
            // Escolhe cluster baseado nos pesos
            double r = (double)rand() / RAND_MAX;
            double acc = 0;
            int chosen = 0;
            
            for(int k = 0; k < num_clusters; k++) {
                acc += weights[k];
                if(r <= acc) {
                    chosen = k;
                    break;
                }
            }
            
            sample_from_cluster(&new_pop[new_size], chosen, scale_factor);
            new_size++;
        }
        
        /* ===== EXPLORAÇÃO LOCAL ===== */
        int local_search = LOCAL_SEARCH;
        for(int i = 0; i < local_search && new_size < MAX_POP; i++) {
            Individual mutated = pop[i % elite_size];
            mutate_individual(&mutated, 0.1 * scale_factor);
            new_pop[new_size++] = mutated;
        }
        
        /* ===== FILTRAGEM ===== */
        int filtered_size = eliminate_weak_and_similar(new_pop, new_size, current_best);
        
        // Garante população mínima
        if(filtered_size < MIN_POP) {
            for(int i = filtered_size; i < MIN_POP && i < MAX_POP; i++) {
                random_individual(&new_pop[i]);
            }
            filtered_size = MIN_POP;
        }
        
        // Atualiza população principal
        for(int i = 0; i < filtered_size; i++)
            pop[i] = new_pop[i];
        pop_size = filtered_size;
        
        /* ===== INJEÇÃO ALEATÓRIA ===== */
        int inject = (int)(RANDOM_RATE * pop_size);
        if(inject < 3) inject = 3;
        
        for(int i = 0; i < inject && pop_size < MAX_POP; i++) {
            random_individual(&pop[pop_size]);
            pop_size++;
        }
        
        // Relatório periódico
        if(gen % 50 == 0 && gen > 0) {
            double avg_fitness = 0;
            for(int i = 0; i < 10 && i < pop_size; i++)
                avg_fitness += pop[i].fitness;
            avg_fitness /= (pop_size < 10 ? pop_size : 10);
            printf("   📊 Média top 10: %.6f\n", avg_fitness);
        }
    }
    
    // Resultado final
    printf("\n========================================\n");
    printf("MELHOR SOLUÇÃO ENCONTRADA:\n");
    printf("x = [");
    for(int i = 0; i < DIM; i++) {
        printf("%.8f", best_individual.x[i]);
        if(i < DIM-1) printf(", ");
    }
    printf("]\n");
    printf("Fitness = %.10f\n", best_fitness_ever);
    printf("Esperado = 0.00000000\n");
    printf("========================================\n");
    
    return 0;
}