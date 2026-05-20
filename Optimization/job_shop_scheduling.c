#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

/* ================= PARÂMETROS ================= */

#define JOBS 10
#define MACHINES 5
#define OPERATIONS (JOBS * MACHINES)

#define POP_MAX 300
#define INIT_POP 150
#define GENERATIONS 2000
#define ELITE_SIZE 15

#define SIM_THRESHOLD 35
#define RANDOM_RATE 0.08
#define LOCAL_SEARCH_RATE 0.3

/* ================= DADOS DO PROBLEMA (LA01) ================= */
int jobs[10][5][2] = {
    {{0, 21}, {1, 53}, {2, 95}, {3, 55}, {4, 34}},
    {{0, 21}, {1, 52}, {2, 16}, {3, 26}, {4, 71}},
    {{0, 39}, {1, 98}, {2, 42}, {3, 31}, {4, 12}},
    {{0, 77}, {1, 55}, {2, 79}, {3, 66}, {4, 77}},
    {{0, 83}, {1, 34}, {2, 64}, {3, 19}, {4, 37}},
    {{0, 54}, {1, 43}, {2, 79}, {3, 92}, {4, 62}},
    {{0, 69}, {1, 77}, {2, 87}, {3, 87}, {4, 93}},
    {{0, 77}, {1, 79}, {2, 52}, {3, 64}, {4, 80}},
    {{0, 19}, {1, 77}, {2, 18}, {3, 69}, {4, 73}},
    {{0, 49}, {1, 66}, {2, 83}, {3, 68}, {4, 26}}
};

/* ================= ESTRUTURAS ================= */

typedef struct {
    int sequence[OPERATIONS];
    int fitness;
} Individual;

/* ================= FUNÇÃO DE AVALIAÇÃO ================= */

int evaluate_sequence(int sequence[]) {
    int job_step[JOBS] = {0};
    int machine_time[MACHINES] = {0};
    int job_time[JOBS] = {0};
    
    for (int i = 0; i < OPERATIONS; i++) {
        int job = sequence[i];
        if (job < 0 || job >= JOBS) return 999999;
        
        int step = job_step[job];
        if (step >= MACHINES) return 999999;
        
        int machine = jobs[job][step][0];
        int duration = jobs[job][step][1];
        
        int start_time = (job_time[job] > machine_time[machine]) ? 
                         job_time[job] : machine_time[machine];
        int end_time = start_time + duration;
        
        job_time[job] = end_time;
        machine_time[machine] = end_time;
        job_step[job]++;
    }
    
    int makespan = 0;
    for (int i = 0; i < JOBS; i++) {
        if (job_time[i] > makespan) makespan = job_time[i];
    }
    for (int i = 0; i < MACHINES; i++) {
        if (machine_time[i] > makespan) makespan = machine_time[i];
    }
    
    return makespan;
}

int fitness(Individual *ind) {
    return evaluate_sequence(ind->sequence);
}

/* ================= GERAÇÃO DE INDIVÍDUOS ================= */

void random_individual(Individual *ind) {
    int remaining[JOBS];
    for (int i = 0; i < JOBS; i++) remaining[i] = MACHINES;
    
    for (int i = 0; i < OPERATIONS; i++) {
        int count = 0;
        for (int j = 0; j < JOBS; j++) {
            if (remaining[j] > 0) count++;
        }
        
        int idx = rand() % count;
        int job = 0;
        int temp = idx;
        for (job = 0; job < JOBS; job++) {
            if (remaining[job] > 0) {
                if (temp == 0) break;
                temp--;
            }
        }
        
        ind->sequence[i] = job;
        remaining[job]--;
    }
    ind->fitness = fitness(ind);
}

/* ================= BUSCA LOCAL MELHORADA ================= */

// Troca duas operações se melhorar
int swap_if_better(Individual *ind, int i, int j) {
    if (i == j) return 0;
    
    int temp = ind->sequence[i];
    ind->sequence[i] = ind->sequence[j];
    ind->sequence[j] = temp;
    
    int new_fitness = fitness(ind);
    if (new_fitness < ind->fitness) {
        ind->fitness = new_fitness;
        return 1;
    } else {
        // Desfaz troca
        temp = ind->sequence[i];
        ind->sequence[i] = ind->sequence[j];
        ind->sequence[j] = temp;
        return 0;
    }
}

// Busca local completa (varredura de swaps)
void local_search(Individual *ind) {
    int improved = 1;
    int iterations = 0;
    
    while (improved && iterations < 100) {
        improved = 0;
        iterations++;
        
        // Tenta trocar todas as posições
        for (int i = 0; i < OPERATIONS - 1; i++) {
            for (int j = i + 1; j < OPERATIONS; j++) {
                if (swap_if_better(ind, i, j)) {
                    improved = 1;
                }
            }
        }
    }
}

/* ================= MATRIZ DE PROBABILIDADES (EDA SIMPLIFICADO) ================= */

double prob_matrix[JOBS][JOBS];  // P(job atual -> próximo job)
double job_probs[JOBS];           // Probabilidade de cada job aparecer

// Aprende modelo de probabilidades (bigrama)
void learn_model(Individual pop[], int size) {
    // Inicializa matriz de bigramas
    for (int i = 0; i < JOBS; i++) {
        for (int j = 0; j < JOBS; j++) {
            prob_matrix[i][j] = 1.0;  // Laplace smoothing
        }
        job_probs[i] = 1.0;
    }
    
    // Conta transições
    for (int k = 0; k < size; k++) {
        for (int i = 0; i < OPERATIONS - 1; i++) {
            int current = pop[k].sequence[i];
            int next = pop[k].sequence[i+1];
            prob_matrix[current][next]++;
        }
        // Conta primeira posição
        job_probs[pop[k].sequence[0]]++;
    }
    
    // Normaliza primeira posição
    double total_jobs = 0;
    for (int i = 0; i < JOBS; i++) {
        total_jobs += job_probs[i];
    }
    for (int i = 0; i < JOBS; i++) {
        job_probs[i] /= total_jobs;
    }
    
    // Normaliza matriz de transição
    for (int i = 0; i < JOBS; i++) {
        double total = 0;
        for (int j = 0; j < JOBS; j++) {
            total += prob_matrix[i][j];
        }
        for (int j = 0; j < JOBS; j++) {
            prob_matrix[i][j] /= total;
        }
    }
}

// Amostra indivíduo usando o modelo de bigramas
void sample_individual(Individual *ind) {
    int remaining[JOBS];
    for (int i = 0; i < JOBS; i++) {
        remaining[i] = MACHINES;
    }
    
    // Amostra primeira posição
    double r = (double)rand() / RAND_MAX;
    double acc = 0;
    int current = 0;
    for (int j = 0; j < JOBS; j++) {
        acc += job_probs[j];
        if (r <= acc) {
            current = j;
            break;
        }
    }
    
    ind->sequence[0] = current;
    remaining[current]--;
    
    // Amostra posições seguintes
    for (int i = 1; i < OPERATIONS; i++) {
        // Calcula probabilidades para o próximo job
        double probs[JOBS];
        double total = 0;
        
        for (int j = 0; j < JOBS; j++) {
            if (remaining[j] > 0) {
                probs[j] = prob_matrix[current][j];
                total += probs[j];
            } else {
                probs[j] = 0;
            }
        }
        
        int next;
        if (total > 0) {
            // Renormaliza
            for (int j = 0; j < JOBS; j++) {
                probs[j] /= total;
            }
            
            r = (double)rand() / RAND_MAX;
            acc = 0;
            next = 0;
            for (int j = 0; j < JOBS; j++) {
                acc += probs[j];
                if (r <= acc) {
                    next = j;
                    break;
                }
            }
        } else {
            // Fallback: escolhe uniformemente
            int count = 0;
            for (int j = 0; j < JOBS; j++) {
                if (remaining[j] > 0) count++;
            }
            int choice = rand() % count;
            next = 0;
            int temp = choice;
            for (next = 0; next < JOBS; next++) {
                if (remaining[next] > 0) {
                    if (temp == 0) break;
                    temp--;
                }
            }
        }
        
        ind->sequence[i] = next;
        remaining[next]--;
        current = next;
    }
    
    ind->fitness = fitness(ind);
}

/* ================= OPERADORES DE MUTAÇÃO ================= */

// Mutação por troca de dois jobs
void swap_mutation(Individual *ind) {
    int pos1 = rand() % OPERATIONS;
    int pos2 = rand() % OPERATIONS;
    
    int temp = ind->sequence[pos1];
    ind->sequence[pos1] = ind->sequence[pos2];
    ind->sequence[pos2] = temp;
    
    ind->fitness = fitness(ind);
}

// Mutação por inserção
void insert_mutation(Individual *ind) {
    int pos1 = rand() % OPERATIONS;
    int pos2 = rand() % OPERATIONS;
    
    if (pos1 == pos2) return;
    
    int temp = ind->sequence[pos1];
    if (pos1 < pos2) {
        for (int i = pos1; i < pos2; i++) {
            ind->sequence[i] = ind->sequence[i+1];
        }
        ind->sequence[pos2] = temp;
    } else {
        for (int i = pos1; i > pos2; i--) {
            ind->sequence[i] = ind->sequence[i-1];
        }
        ind->sequence[pos2] = temp;
    }
    
    ind->fitness = fitness(ind);
}

// Mutação por inversão
void invert_mutation(Individual *ind) {
    int pos1 = rand() % OPERATIONS;
    int pos2 = rand() % OPERATIONS;
    
    if (pos1 > pos2) {
        int temp = pos1;
        pos1 = pos2;
        pos2 = temp;
    }
    
    for (int i = 0; i < (pos2 - pos1 + 1) / 2; i++) {
        int temp = ind->sequence[pos1 + i];
        ind->sequence[pos1 + i] = ind->sequence[pos2 - i];
        ind->sequence[pos2 - i] = temp;
    }
    
    ind->fitness = fitness(ind);
}

/* ================= UTIL ================= */

int cmp(const void *a, const void *b) {
    return ((Individual*)a)->fitness - ((Individual*)b)->fitness;
}

double euclidean_distance(Individual *a, Individual *b) {
    double sum = 0;
    for (int i = 0; i < OPERATIONS; i++) {
        double diff = a->sequence[i] - b->sequence[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

/* ================= MAIN ================= */

int main() {
    srand(time(NULL));
    
    printf("Job Shop Scheduling Problem - LA01\n");
    printf("EDA with Bigram Model + Local Search\n");
    printf("Jobs: %d, Machines: %d, Operations: %d\n", JOBS, MACHINES, OPERATIONS);
    printf("Optimal makespan: 666\n\n");
    
    Individual *pop = malloc(POP_MAX * sizeof(Individual));
    int pop_size = INIT_POP;
    int best_fitness_ever = 1e9;
    Individual best_solution;
    
    // Inicializa população
    printf("Gerando população inicial...\n");
    for (int i = 0; i < pop_size; i++) {
        random_individual(&pop[i]);
        if (pop[i].fitness < best_fitness_ever) {
            best_fitness_ever = pop[i].fitness;
            best_solution = pop[i];
        }
        if ((i+1) % 50 == 0) printf(".");
    }
    printf("\nMelhor inicial: %d\n\n", best_fitness_ever);
    
    int no_improvement = 0;
    int last_best = best_fitness_ever;
    
    for (int g = 0; g < GENERATIONS; g++) {
        
        // Avalia
        for (int i = 0; i < pop_size; i++) {
            pop[i].fitness = fitness(&pop[i]);
        }
        
        qsort(pop, pop_size, sizeof(Individual), cmp);
        
        // Atualiza melhor
        if (pop[0].fitness < best_fitness_ever) {
            best_fitness_ever = pop[0].fitness;
            best_solution = pop[0];
            
            double gap = (best_fitness_ever - 666) * 100.0 / 666;
            printf("Gen %5d | Best: %d | Gap: %.2f%% | Pop: %d\n", 
                   g, best_fitness_ever, gap, pop_size);
            
            no_improvement = 0;
            last_best = best_fitness_ever;
        } else {
            no_improvement++;
        }
        
        // Critério de parada
        if (best_fitness_ever <= 666) {
            printf("\n*** SOLUÇÃO ÓTIMA ENCONTRADA! ***\n");
            break;
        }
        
        if (no_improvement > 300) {
            printf("\nEstagnação, reinjetando aleatórios...\n");
            // Reinjeta muitos aleatórios
            for (int i = 0; i < pop_size / 2; i++) {
                random_individual(&pop[i]);
            }
            no_improvement = 0;
        }
        
        // Preserva elite
        Individual elite[ELITE_SIZE];
        int elite_count = 0;
        for (int i = 0; i < ELITE_SIZE && i < pop_size; i++) {
            elite[elite_count++] = pop[i];
        }
        
        // Aprende modelo (a cada 5 gerações)
        if (g % 5 == 0 && pop_size >= 50) {
            learn_model(pop, pop_size);
        }
        
        // Gera novos indivíduos
        Individual *new_pop = malloc(POP_MAX * sizeof(Individual));
        int new_size = 0;
        
        // Adiciona elite
        for (int i = 0; i < elite_count; i++) {
            new_pop[new_size++] = elite[i];
        }
        
        // Gera filhos por amostragem
        int to_generate = POP_MAX - (int)(RANDOM_RATE * POP_MAX);
        while (new_size < to_generate) {
            Individual new_ind;
            
            // 70% das vezes usa amostragem, 30% usa mutação de elite
            if (rand() % 100 < 70 && g > 10) {
                sample_individual(&new_ind);
            } else {
                // Copia um indivíduo da elite e aplica mutação
                int idx = rand() % elite_count;
                memcpy(new_ind.sequence, elite[idx].sequence, sizeof(new_ind.sequence));
                
                int mut_type = rand() % 3;
                if (mut_type == 0) swap_mutation(&new_ind);
                else if (mut_type == 1) insert_mutation(&new_ind);
                else invert_mutation(&new_ind);
            }
            
            // Aplica busca local em indivíduos promissores
            if (new_ind.fitness < best_fitness_ever + 100 && 
                (double)rand() / RAND_MAX < LOCAL_SEARCH_RATE) {
                local_search(&new_ind);
            }
            
            new_pop[new_size++] = new_ind;
        }
        
        // Adiciona aleatórios
        int random_count = (int)(RANDOM_RATE * POP_MAX);
        for (int i = 0; i < random_count && new_size < POP_MAX; i++) {
            random_individual(&new_pop[new_size]);
            new_size++;
        }
        
        // Remove similares (distância euclidiana)
        Individual *temp = malloc(POP_MAX * sizeof(Individual));
        int tsize = 0;
        
        for (int i = 0; i < new_size; i++) {
            int keep = 1;
            for (int j = 0; j < tsize; j++) {
                if (euclidean_distance(&new_pop[i], &temp[j]) < SIM_THRESHOLD) {
                    keep = 0;
                    break;
                }
            }
            if (keep && tsize < POP_MAX) {
                temp[tsize++] = new_pop[i];
            }
        }
        
        free(pop);
        pop = malloc(POP_MAX * sizeof(Individual));
        pop_size = tsize;
        for (int i = 0; i < pop_size; i++) {
            pop[i] = temp[i];
        }
        free(temp);
        free(new_pop);
        
        // Mantém população mínima
        if (pop_size < INIT_POP / 2) {
            for (int i = pop_size; i < INIT_POP; i++) {
                random_individual(&pop[i]);
            }
            pop_size = INIT_POP;
        }
    }
    
    // Avaliação final com busca local na melhor solução
    printf("\nAplicando busca local na melhor solução...\n");
    local_search(&best_solution);
    best_fitness_ever = best_solution.fitness;
    
    printf("\n=== RESULTADO FINAL ===\n");
    printf("Makespan: %d\n", best_fitness_ever);
    printf("Ótimo conhecido: 666\n");
    printf("Gap: %.2f%%\n", (best_fitness_ever - 666) * 100.0 / 666);
    
    if (best_fitness_ever <= 666) {
        printf("✅ SOLUÇÃO ÓTIMA ENCONTRADA!\n");
    } else if (best_fitness_ever <= 680) {
        printf("👍 SOLUÇÃO MUITO BOA!\n");
    } else if (best_fitness_ever <= 720) {
        printf("📈 SOLUÇÃO RAZOÁVEL\n");
    } else {
        printf("⚠️  SOLUÇÃO PODE SER MELHORADA\n");
    }
    
    free(pop);
    return 0;
}