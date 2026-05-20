#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* ================= PARÂMETROS ================= */

#define N 8
#define POP_MAX 400
#define INIT_POP 100
#define GENERATIONS 500  // Aumentado para melhor convergência
#define ELITE_SIZE 5

#define SIM_THRESHOLD 3   // Reduzido para TSP (permutações)
#define RANDOM_RATE 0.05

/* ================= MATRIZ DE DISTÂNCIAS ================= */

double dist[N][N] = {
    {0,2,9,10,7,3,8,6},
    {2,0,8,5,6,4,3,7},
    {9,8,0,4,3,7,2,5},
    {10,5,4,0,3,6,7,8},
    {7,6,3,3,0,4,5,6},
    {3,4,7,6,4,0,2,3},
    {8,3,2,7,5,2,0,4},
    {6,7,5,8,6,3,4,0}
};

/* ================= ESTRUTURA ================= */

typedef struct {
    int x[N];
    double fitness;
} Individual;

/* ================= MODELO EDA ================= */

double prob[N][N]; // P(j | i)

/* ================= FITNESS ================= */

double fitness(Individual *ind) {
    double cost = 0;

    for (int i = 0; i < N - 1; i++)
        cost += dist[ind->x[i]][ind->x[i+1]];

    cost += dist[ind->x[N-1]][ind->x[0]];

    return cost;
}

/* ================= UTIL ================= */

void random_individual(Individual *ind) {
    for (int i = 0; i < N; i++)
        ind->x[i] = i;

    // Fisher-Yates shuffle
    for (int i = N - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = ind->x[i];
        ind->x[i] = ind->x[j];
        ind->x[j] = tmp;
    }

    ind->fitness = fitness(ind);
}

int cmp(const void *a, const void *b) {
    double diff = ((Individual*)a)->fitness - ((Individual*)b)->fitness;
    return (diff > 0) - (diff < 0);
}

/* ================= DISTÂNCIA ================= */

int hamming(Individual *a, Individual *b) {
    int d = 0;
    for (int i = 0; i < N; i++)
        if (a->x[i] != b->x[i]) d++;
    return d;
}

/* ================= APRENDIZADO CORRIGIDO ================= */

void learn_model(Individual pop[], int size) {
    // Inicializa com Laplace smoothing
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            prob[i][j] = 1.0;

    // Conta transições
    for (int k = 0; k < size; k++) {
        for (int i = 0; i < N - 1; i++) {
            int a = pop[k].x[i];
            int b = pop[k].x[i+1];
            prob[a][b]++;
        }
        // Última transição (volta ao início)
        int a = pop[k].x[N-1];
        int b = pop[k].x[0];
        prob[a][b]++;
    }

    // Normaliza por linha
    for (int i = 0; i < N; i++) {
        double sum = 0;
        for (int j = 0; j < N; j++)
            sum += prob[i][j];
        
        if (sum > 0) {
            for (int j = 0; j < N; j++)
                prob[i][j] /= sum;
        } else {
            // Fallback: distribuição uniforme
            for (int j = 0; j < N; j++)
                prob[i][j] = 1.0 / N;
        }
    }
}

/* ================= SAMPLE CORRIGIDO ================= */

int select_next(int current, int used[]) {
    // Calcula probabilidades condicionais apenas para não visitados
    double total_prob = 0;
    double probs[N];
    
    for (int j = 0; j < N; j++) {
        if (used[j]) {
            probs[j] = 0;
        } else {
            probs[j] = prob[current][j];
            total_prob += probs[j];
        }
    }
    
    if (total_prob == 0) {
        // Fallback: escolhe uniformemente entre não visitados
        int count = 0;
        for (int j = 0; j < N; j++)
            if (!used[j]) count++;
        
        int idx = rand() % count;
        for (int j = 0; j < N; j++) {
            if (!used[j]) {
                if (idx == 0) return j;
                idx--;
            }
        }
    }
    
    // Seleção baseada em probabilidades renormalizadas
    double r = (double)rand() / RAND_MAX;
    double acc = 0;
    
    for (int j = 0; j < N; j++) {
        if (used[j]) continue;
        double norm_prob = probs[j] / total_prob;
        acc += norm_prob;
        if (r <= acc) return j;
    }
    
    // Fallback seguro
    for (int j = 0; j < N; j++)
        if (!used[j]) return j;
    
    return 0;
}

void sample_individual(Individual *ind) {
    int used[N] = {0};
    
    // Escolhe cidade inicial aleatória
    int current = rand() % N;
    ind->x[0] = current;
    used[current] = 1;
    
    // Constrói o tour
    for (int i = 1; i < N; i++) {
        int next = select_next(current, used);
        ind->x[i] = next;
        used[next] = 1;
        current = next;
    }
    
    ind->fitness = fitness(ind);
}

/* ================= MAIN CORRIGIDO ================= */

int main() {
    srand(time(NULL));
    
    Individual pop[POP_MAX];
    int pop_size = INIT_POP;
    double best_fitness = 1e9;
    Individual best_solution;
    
    // Inicializa
    for (int i = 0; i < pop_size; i++)
        random_individual(&pop[i]);
    
    for (int g = 0; g < GENERATIONS; g++) {
        
        // Avalia
        for (int i = 0; i < pop_size; i++)
            pop[i].fitness = fitness(&pop[i]);
        
        qsort(pop, pop_size, sizeof(Individual), cmp);
        
        // Atualiza melhor solução global
        if (pop[0].fitness < best_fitness) {
            best_fitness = pop[0].fitness;
            best_solution = pop[0];
        }
        
        printf("Gen %3d | Best: %.2f | Global Best: %.2f | Pop: %d\n",
               g, pop[0].fitness, best_fitness, pop_size);
        
        // Critério de parada antecipada
        // if (best_fitness <= 26.0) {
        //     printf("\nOptimal or near-optimal solution found!\n");
        //     break;
        // }
        
        // Preserva elite
        Individual elite[ELITE_SIZE];
        int elite_count = 0;
        for (int i = 0; i < ELITE_SIZE && i < pop_size; i++)
            elite[elite_count++] = pop[i];
        
        // Aprende modelo
        learn_model(pop, pop_size);
        
        // Amostra novos indivíduos
        int new_size = pop_size;
        int samples = POP_MAX - pop_size;
        for (int i = 0; i < samples; i++) {
            Individual new_ind;
            sample_individual(&new_ind);
            if (new_size < POP_MAX) {
                pop[new_size++] = new_ind;
            }
        }
        pop_size = new_size;
        
        // Remove similares (preservando elite)
        Individual temp[POP_MAX];
        int tsize = 0;
        
        // Primeiro adiciona elite
        for (int i = 0; i < elite_count; i++)
            temp[tsize++] = elite[i];
        
        // Depois adiciona outros não similares
        for (int i = 0; i < pop_size; i++) {
            int keep = 1;
            for (int j = 0; j < tsize; j++) {
                if (hamming(&pop[i], &temp[j]) < SIM_THRESHOLD) {
                    keep = 0;
                    break;
                }
            }
            if (keep && tsize < POP_MAX) {
                temp[tsize++] = pop[i];
            }
        }
        
        pop_size = tsize;
        for (int i = 0; i < pop_size; i++)
            pop[i] = temp[i];
        
        // Injeção aleatória
        int inject = (int)(RANDOM_RATE * POP_MAX);
        for (int i = 0; i < inject && pop_size < POP_MAX; i++) {
            random_individual(&pop[pop_size]);
            pop_size++;
        }
        
        // Evita colapso
        if (pop_size < INIT_POP / 2) {
            for (int i = pop_size; i < INIT_POP; i++)
                random_individual(&pop[i]);
            pop_size = INIT_POP;
        }
    }
    
    // Avaliação final
    for (int i = 0; i < pop_size; i++)
        pop[i].fitness = fitness(&pop[i]);
    qsort(pop, pop_size, sizeof(Individual), cmp);
    
    if (pop[0].fitness < best_fitness) {
        best_fitness = pop[0].fitness;
        best_solution = pop[0];
    }
    
    printf("\nBest tour found:\n");
    for (int i = 0; i < N; i++)
        printf("%d ", best_solution.x[i]);
    printf("\nTotal cost = %.2f\n", best_fitness);
    
    return 0;
}