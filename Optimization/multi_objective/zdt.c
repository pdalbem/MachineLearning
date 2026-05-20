#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define D 30                    // Dimensão do problema
#define INIT_POP 50
#define MAX_GEN 200
#define MAX_CLUSTERS 5
#define SIMILARITY_THRESHOLD 1e-3
#define NEW_INDIVIDUALS_RATIO 0.05

typedef struct {
    double x[D];
    double f1, f2;
} Individual;

typedef struct {
    Individual *inds;
    int size;
} Population;

/* ================= DISTÂNCIA EUCLIDIANA ================= */
double distance(Individual *a, Individual *b) {
    double sum = 0;
    for (int i = 0; i < D; i++) sum += (a->x[i] - b->x[i]) * (a->x[i] - b->x[i]);
    return sqrt(sum);
}

/* ================= FUNÇÕES ZDT ================= */
void evaluate_zdt(Individual *ind, int problem) {
    double sum = 0;
    for (int i = 1; i < D; i++) sum += ind->x[i];

    double g, h;
    switch(problem) {
        case 1: // ZDT1
            ind->f1 = ind->x[0];
            g = 1.0 + 9.0 * sum / (D - 1);
            h = 1.0 - sqrt(ind->f1 / g);
            ind->f2 = g * h;
            break;
        case 2: // ZDT2
            ind->f1 = ind->x[0];
            g = 1.0 + 9.0 * sum / (D - 1);
            h = 1.0 - pow(ind->f1 / g, 2.0);
            ind->f2 = g * h;
            break;
        case 3: // ZDT3
            ind->f1 = ind->x[0];
            g = 1.0 + 9.0 * sum / (D - 1);
            h = 1.0 - sqrt(ind->f1 / g) - (ind->f1 / g) * sin(10 * M_PI * ind->x[0]);
            ind->f2 = g * h;
            break;
        case 4: // ZDT4
            sum = 0;
            for (int i = 1; i < D; i++) sum += ind->x[i]*ind->x[i] - 10*cos(4*M_PI*ind->x[i]);
            ind->f1 = ind->x[0];
            g = 1.0 + 10*(D-1) + sum;
            h = 1.0 - sqrt(ind->f1 / g);
            ind->f2 = g * h;
            break;
        case 5: // ZDT5 (discreto binário, aqui aproximado contínuo)
            ind->f1 = ind->x[0];
            g = 1.0 + 9.0 * sum / (D-1);
            h = 1.0 - sqrt(ind->f1 / g);
            ind->f2 = g * h;
            break;
        case 6: // ZDT6
            ind->f1 = 1 - exp(-4*ind->x[0]) * pow(sin(6*M_PI*ind->x[0]),6);
            g = 1.0 + 9.0 * pow(sum / (D-1), 0.25);
            h = 1.0 - pow(ind->f1 / g,2);
            ind->f2 = g * h;
            break;
        default:
            ind->f1 = ind->x[0]; ind->f2 = sum;
    }
}

/* ================= PARETO ================= */
int dominates(Individual *a, Individual *b) {
    return ((a->f1 <= b->f1 && a->f2 <= b->f2) && (a->f1 < b->f1 || a->f2 < b->f2));
}

void pareto_filter(Population *pop) {
    int i = 0;
    while (i < pop->size) {
        int dominated = 0;
        for (int j = 0; j < pop->size; j++)
            if (i != j && dominates(&pop->inds[j], &pop->inds[i])) dominated = 1;
        if (dominated) {
            for (int k = i; k < pop->size-1; k++) pop->inds[k] = pop->inds[k+1];
            pop->size--;
        } else i++;
    }
}

void remove_similar(Population *pop) {
    for (int i = 0; i < pop->size; i++) {
        for (int j = i+1; j < pop->size; ) {
            if (distance(&pop->inds[i], &pop->inds[j]) < SIMILARITY_THRESHOLD) {
                for (int k = j; k < pop->size-1; k++) pop->inds[k] = pop->inds[k+1];
                pop->size--;
            } else j++;
        }
    }
}

/* ================= INICIALIZAÇÃO ================= */
void init_population(Population *pop) {
    pop->size = INIT_POP;
    pop->inds = (Individual*)malloc(pop->size * sizeof(Individual));
    for (int i = 0; i < pop->size; i++) {
        for (int d = 0; d < D; d++) pop->inds[i].x[d] = ((double)rand()/RAND_MAX);
        evaluate_zdt(&pop->inds[i], 1);
    }
}

/* ================= KMEANS + GMM ================= */
void kmeans(Individual *inds, int n, int clusters, double centers[clusters][D], int assignments[n]) {
    for (int c=0; c<clusters; c++) {
        int idx = rand()%n;
        for (int d=0; d<D; d++) centers[c][d] = inds[idx].x[d];
    }

    int changed = 1;
    while(changed) {
        changed=0;
        for(int i=0;i<n;i++) {
            int best=0;
            double best_dist=0;
            for(int d=0;d<D;d++) best_dist += (inds[i].x[d]-centers[0][d])*(inds[i].x[d]-centers[0][d]);
            for(int c=1;c<clusters;c++) {
                double dist=0;
                for(int d=0;d<D;d++) dist += (inds[i].x[d]-centers[c][d])*(inds[i].x[d]-centers[c][d]);
                if(dist<best_dist){best=c; best_dist=dist;}
            }
            if(assignments[i]!=best){assignments[i]=best; changed=1;}
        }
        for(int c=0;c<clusters;c++){
            double sum[D]={0};
            int count=0;
            for(int i=0;i<n;i++) if(assignments[i]==c){for(int d=0;d<D;d++) sum[d]+=inds[i].x[d]; count++;}
            if(count>0) for(int d=0;d<D;d++) centers[c][d]=sum[d]/count;
        }
    }
}

void sample_gmm(Population *pop, Individual *out, int problem) {
    int clusters = (pop->size<MAX_CLUSTERS)?pop->size:MAX_CLUSTERS;
    int assignments[pop->size];
    double centers[clusters][D];
    kmeans(pop->inds,pop->size,clusters,centers,assignments);

    int c = rand()%clusters;
    double stddev[D]={0};
    int count=0;
    for(int i=0;i<pop->size;i++)
        if(assignments[i]==c){for(int d=0;d<D;d++) stddev[d]+=(pop->inds[i].x[d]-centers[c][d])*(pop->inds[i].x[d]-centers[c][d]); count++;}
    if(count>0) for(int d=0;d<D;d++) stddev[d]=sqrt(stddev[d]/count);

    for(int d=0;d<D;d++)
        out->x[d]=centers[c][d]+stddev[d]*((double)rand()/RAND_MAX-0.5)*2.0;

    evaluate_zdt(out, problem);
}

/* ================= INSERE NOVOS INDIVÍDUOS ALEATÓRIOS ================= */
void insert_new(Population *pop, int problem) {
    int n_new = (int)(pop->size * NEW_INDIVIDUALS_RATIO);
    pop->inds = (Individual*)realloc(pop->inds, (pop->size+n_new)*sizeof(Individual));
    for(int i=0;i<n_new;i++){
        for(int d=0;d<D;d++) pop->inds[pop->size+i].x[d]=((double)rand()/RAND_MAX);
        evaluate_zdt(&pop->inds[pop->size+i], problem);
    }
    pop->size+=n_new;
}

/* ================= MAIN ================= */
int main() {
    srand(time(NULL));
    int problem = 1; // ZDT1-ZDT6
    Population pop;
    init_population(&pop);

    for(int gen=0;gen<MAX_GEN;gen++){
        pareto_filter(&pop);
        remove_similar(&pop);

        int orig_size = pop.size;
        for(int i=0;i<orig_size;i++){
            Individual new_ind;
            sample_gmm(&pop,&new_ind,problem);
            pop.inds = (Individual*)realloc(pop.inds,(pop.size+1)*sizeof(Individual));
            pop.inds[pop.size++] = new_ind;
        }

        insert_new(&pop, problem);

        if(pop.size<20){
            int extra = 20 - pop.size;
            pop.inds = (Individual*)realloc(pop.inds,(pop.size+extra)*sizeof(Individual));
            for(int i=0;i<extra;i++){
                pop.inds[pop.size+i].x[0]=((double)rand()/RAND_MAX);
                evaluate_zdt(&pop.inds[pop.size+i],problem);
            }
            pop.size+=extra;
        }

        printf("Gen %3d | Pareto size: %3d | Pop: %3d\n", gen,pop.size,pop.size);
    }

    printf("\nPareto front:\n");
    for(int i=0;i<pop.size;i++)
        printf("f1=%.6f f2=%.6f\n", pop.inds[i].f1, pop.inds[i].f2);

    free(pop.inds);
    return 0;
}