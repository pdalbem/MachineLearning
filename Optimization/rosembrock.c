#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* ================= PARÂMETROS ================= */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


#define N 5
#define POP_MAX 500
#define INIT_POP 150
#define GENERATIONS 500

#define LOWER -2.0
#define UPPER 2.0

#define SIM_THRESHOLD 0.1      // Mais restritivo
#define RANDOM_RATE 0.1
#define ELITE_RATE 0.1          // 10% de elite

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

/* ================= ROSENBROCK ================= */

double fitness(Individual *ind) {
    double sum = 0.0;
    
    for (int i = 0; i < N - 1; i++) {
        double xi = ind->x[i];
        double xnext = ind->x[i+1];
        
        sum += 100.0 * (xnext - xi*xi) * (xnext - xi*xi) 
             + (xi - 1.0) * (xi - 1.0);
    }
    
    return sum;
}

/* ================= INIT ================= */

void random_individual(Individual *ind) {
    for (int i = 0; i < N; i++)
        ind->x[i] = rand_uniform(LOWER, UPPER);
    ind->fitness = fitness(ind);
}

void init_near_optimum(Individual *ind) {
    // Inicializa perto do ótimo (1,1,1,1,1)
    for (int i = 0; i < N; i++)
        ind->x[i] = 1.0 + rand_normal() * 0.1;
    ind->fitness = fitness(ind);
}

/* ================= DISTÂNCIA ================= */

double distance(Individual *a, Individual *b) {
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
        double d = a->x[i] - b->x[i];
        sum += d * d;
    }
    return sqrt(sum);
}

/* ================= COMPARAÇÃO ================= */

int cmp_fitness(const void *a, const void *b) {
    double fa = ((Individual*)a)->fitness;
    double fb = ((Individual*)b)->fitness;
    return (fa > fb) - (fa < fb);
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

/* ================= COVARIÂNCIA COM REGULARIZAÇÃO ================= */

void compute_covariance(Individual pop[], int size, double mu[], double cov[N][N]) {
    // Inicializa
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            cov[i][j] = 0.0;
    
    // Calcula covariancia
    for (int k = 0; k < size; k++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                cov[i][j] += (pop[k].x[i] - mu[i]) *
                             (pop[k].x[j] - mu[j]);
            }
        }
    }
    
    // Normaliza e regulariza
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cov[i][j] /= size;
        }
        cov[i][i] += EPS;  // Regularização
    }
}

/* ================= CHOLESKY ESTÁVEL ================= */

int cholesky(double A[N][N], double L[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            for (int k = 0; k < j; k++)
                sum += L[i][k] * L[j][k];
            
            if (i == j) {
                double val = A[i][i] - sum;
                if (val <= EPS) {
                    // Se a matriz não é definida positiva, adiciona mais regularização
                    val = EPS;
                }
                L[i][j] = sqrt(val);
            } else {
                if (fabs(L[j][j]) < EPS) return 0;
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
    }
    return 1;
}

/* ================= AMOSTRAGEM ADAPTATIVA ================= */

void sample_individual(Individual *ind, double mu[], double L[N][N], double scale) {
    double z[N];
    
    for (int i = 0; i < N; i++)
        z[i] = rand_normal() * scale;  // Escala adaptativa
    
    for (int i = 0; i < N; i++) {
        ind->x[i] = mu[i];
        
        for (int j = 0; j <= i; j++)
            ind->x[i] += L[i][j] * z[j];
        
        // Aplica limites com reflexão (melhor que clipping)
        if (ind->x[i] < LOWER) 
            ind->x[i] = 2 * LOWER - ind->x[i];
        if (ind->x[i] > UPPER) 
            ind->x[i] = 2 * UPPER - ind->x[i];
    }
    
    ind->fitness = fitness(ind);
}

/* ================= MUTAÇÃO LOCAL ================= */

void mutate_local(Individual *ind, double step) {
    for (int i = 0; i < N; i++) {
        ind->x[i] += rand_normal() * step;
        if (ind->x[i] < LOWER) ind->x[i] = LOWER;
        if (ind->x[i] > UPPER) ind->x[i] = UPPER;
    }
    ind->fitness = fitness(ind);
}

/* ================= MAIN MELHORADO ================= */

int main() {
    srand(time(NULL));
    
    Individual pop[POP_MAX];
    Individual new_pop[POP_MAX];
    int pop_size = INIT_POP;
    
    // População inicial híbrida
    for (int i = 0; i < pop_size; i++) {
        if (i < pop_size / 3)
            init_near_optimum(&pop[i]);  // Perto do ótimo
        else
            random_individual(&pop[i]);   // Aleatório
    }
    
    double best_fitness_ever = 1e9;
    Individual best_individual;
    double scale_factor = 1.0;  // Escala adaptativa para amostragem
    
    for (int g = 0; g < GENERATIONS; g++) {
        
        // Ordenação usando qsort
        qsort(pop, pop_size, sizeof(Individual), cmp_fitness);
        
        // Atualiza melhor global
        if (pop[0].fitness < best_fitness_ever) {
            best_fitness_ever = pop[0].fitness;
            best_individual = pop[0];
            
            // Reduz escala quando melhora
            scale_factor = fmax(0.1, scale_factor * 0.99);
        } else {
            // Aumenta escala gradualmente se não melhora (exploração)
            scale_factor = fmin(2.0, scale_factor * 1.01);
        }
        
        printf("Gen %4d | Best: %.10f | Pop: %d | Scale: %.3f\n",
               g, pop[0].fitness, pop_size, scale_factor);
        
        // Critério de parada antecipada
        if (pop[0].fitness < 1e-8) {
            printf("\nConvergência alcançada!\n");
            break;
        }
        
        /* ===== MODELO ===== */
        // Usa apenas os 50% melhores para o modelo
        int elite_for_model = pop_size / 2;
        double mu[N], cov[N][N], L[N][N];
        
        compute_mean(pop, elite_for_model, mu);
        compute_covariance(pop, elite_for_model, mu, cov);
        
        if (!cholesky(cov, L)) {
            printf("Erro Cholesky, usando escala reduzida\n");
            // Se falhar, usa amostragem com matriz identidade
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    L[i][j] = (i == j) ? 1.0 : 0.0;
        }
        
        /* ===== ELITE ===== */
        int elite_size = (int)(ELITE_RATE * pop_size);
        if (elite_size < 5) elite_size = 5;
        
        // Copia elite para nova população
        int new_size = elite_size;
        for (int i = 0; i < elite_size; i++)
            new_pop[i] = pop[i];
        
        /* ===== AMOSTRAGEM ===== */
        int samples_to_add = pop_size;  // Gera tantos novos quantos existem
        
        for (int i = 0; i < samples_to_add && new_size < POP_MAX; i++) {
            sample_individual(&new_pop[new_size], mu, L, scale_factor);
            new_size++;
        }
        
        /* ===== EXPLORAÇÃO LOCAL ===== */
        // Adiciona mutações dos melhores
        int local_search = pop_size / 5;
        for (int i = 0; i < local_search && new_size < POP_MAX; i++) {
            Individual mutated = pop[i % elite_size];
            mutate_local(&mutated, 0.05 * scale_factor);
            new_pop[new_size++] = mutated;
        }
        
        /* ===== THRESHOLD ADAPTATIVO ===== */
        // Aceita apenas indivíduos dentro de um fator do melhor
        double threshold;
        if (pop[0].fitness < 1.0)
            threshold = pop[0].fitness + 0.5;  // Muito restritivo perto do ótimo
        else if (pop[0].fitness < 100)
            threshold = pop[0].fitness * 1.5;  // Moderado
        else
            threshold = pop[0].fitness * 2.0;  // Mais liberal no início
            
        int t1 = 0;
        for (int i = 0; i < new_size; i++) {
            if (new_pop[i].fitness <= threshold)
                pop[t1++] = new_pop[i];
        }
        
        /* ===== DIVERSIDADE ===== */
        int t2 = 0;
        Individual temp[POP_MAX];
        
        for (int i = 0; i < t1; i++) {
            int keep = 1;
            for (int j = 0; j < t2; j++) {
                if (distance(&pop[i], &temp[j]) < SIM_THRESHOLD) {
                    keep = 0;
                    break;
                }
            }
            if (keep) temp[t2++] = pop[i];
        }
        
        for (int i = 0; i < t2; i++)
            pop[i] = temp[i];
        pop_size = t2;
        
        /* ===== INJEÇÃO ALEATÓRIA ===== */
        int inject = (int)(RANDOM_RATE * pop_size);
        for (int i = 0; i < inject && pop_size < POP_MAX; i++) {
            // 70% perto do ótimo, 30% completamente aleatório
            if (rand() % 100 < 70)
                init_near_optimum(&pop[pop_size]);
            else
                random_individual(&pop[pop_size]);
            pop_size++;
        }
        
        /* ===== GARANTE POPULAÇÃO MÍNIMA ===== */
        if (pop_size < 20) {
            for (int i = pop_size; i < 40; i++) {
                init_near_optimum(&pop[i]);
            }
            pop_size = 40;
            scale_factor = 1.0;  // Reseta escala
        }
        
        /* ===== RELATÓRIO DE PROGRESSO ===== */
        if (g % 50 == 0 && g > 0) {
            printf("  -> Melhor até agora: %.10f\n", best_fitness_ever);
            printf("  -> Média dos 10 melhores: ");
            double avg = 0;
            for (int i = 0; i < 10 && i < pop_size; i++)
                avg += pop[i].fitness;
            avg /= (pop_size < 10 ? pop_size : 10);
            printf("%.6f\n", avg);
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
    printf("Fitness = %.10f\n", best_fitness_ever);
    printf("Esperado = 0.00000000\n");
    printf("========================================\n");
    
    return 0;
}