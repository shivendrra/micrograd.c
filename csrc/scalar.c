#include "scalar.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void noop_backward(Scalar *v) {}

Scalar* initialize_scalars(double data, Scalar** child, int child_size) {
  Scalar* v = (Scalar*)malloc(sizeof(Scalar));
  v->data = data;
  v->grad = 0.0;
  v->_prev_size = child_size;
  v->_prev = child;
  v->_backward = noop_backward;
  v->aux = NULL;
  return v;
}

void add_backward(Scalar* v) {
  Scalar* a = v->_prev[0];
  Scalar* b = v->_prev[1];

  a->grad += v->grad;
  b->grad += v->grad;
}

Scalar* add_val(Scalar* a, Scalar* b) {
  Scalar **child = (Scalar**)malloc(2 * sizeof(Scalar*));
  child[0] = a;
  child[1] = b;

  Scalar* out = initialize_scalars(a->data + b->data, child, 2);
  out->_backward = add_backward;
  return out;
}

void mul_backward(Scalar* v) {
  Scalar* a = v->_prev[0];
  Scalar* b = v->_prev[1];

  a->grad += v->grad * b->data;
  b->grad += v->grad * a->data;
}

Scalar* mul_val(Scalar* a, Scalar* b) {
  Scalar **child = (Scalar**)malloc(2 * sizeof(Scalar*));
  child[0] = a;
  child[1] = b;

  Scalar* out = initialize_scalars(a->data * b->data, child, 2);
  out->_backward = mul_backward;
  return out;
}

void pow_backward(Scalar* v) {
  Scalar* a = v->_prev[0];

  a->grad += v->grad * (v->aux * pow(a->data, v->exp-1));
}

Scalar* pow_val(Scalar* a, float exp) {
  Scalar **child = (Scalar**)malloc(1 * sizeof(Scalar*));
  child[0] = a;

  Scalar* out = initialize_scalars(pow(a->data, exp), child, 1);
  out->aux = exp;
  out->_backward = pow_backward;
  return out;
}

void relu_backward(Scalar* v) {
  Scalar* a = v->_prev[0];

  a->grad += v->grad * (v->data > 0);
}

Scalar* relu(Scalar* a) {
  Scalar **child = (Scalar**)malloc(1 * sizeof(Scalar*));
  child[0] = a;
  
  Scalar* out = initialize_scalars(a->data > 0 ? a->data : 0, child, 1);
  out->_backward = relu_backward;
  return out;
}

void tanh_backward(Scalar* v) {
  Scalar* a = v->_prev[0];

  a->grad += v->grad * (1 - pow(a->data, 2));
}

Scalar* tan_h(Scalar* a) {
  Scalar** child = (Scalar**)malloc(1 * sizeof(Scalar*));
  child[0] = a;

  Scalar* out = initialize_scalars(tanh(a->data), child, 1);
  out->_backward = tanh_backward;
  return out;
}

void sigmoid_backward(Scalar* v) {
  Scalar* a = v->_prev[0];

  a->grad += v->grad * (a->data * (1 - a->data));
}

Scalar* sigmoid(Scalar* a) {
  Scalar** child = (Scalar**)malloc(1 * sizeof(Scalar*));
  child[0] = a;

  Scalar* out = initialize_scalars(1 / (1 + exp(-a->data)), child, 1);
  out->_backward = sigmoid_backward;
  return out;
}

void silu_backward(Scalar* v) {
  Scalar* a = v->_prev[0];

  a->grad += v->grad * (a->data * (1 - a->data));
}

Scalar* silu(Scalar* a) {
  Scalar** child = (Scalar**)malloc(1 * sizeof(Scalar*));
  child[0] = a;

  Scalar* out = initialize_scalars(mul_val(a, sigmoid(a)), child, 1);
  out->_backward = silu_backward;
  return out;
}

Scalar* negate(Scalar* a) {
  return mul_val(a, initialize_scalars(-1, NULL, 0));
}

Scalar* sub_val(Scalar* a, Scalar* b) {
  return add_val(a, negate(b));
}

void build_topo(Scalar* v, Scalar*** topo, int* topo_size, Scalar*** visited, int* visited_size) {
  for (int i = 0; i < *visited_size; ++i) {
    if((*visited)[i] == v) return;
  }
  (*visited)[(*visited_size)++] = v;
  for(int i = 0; i < v->_prev_size; ++i) {
    build_topo(v->_prev[i], topo, topo_size, visited, visited_size);
  }
  (*topo)[(*topo_size)++] = v;
}

void backward(Scalar* v) {
  int topo_size = 0;
  int visited_size = 0;

  Scalar** topo = (Scalar**)malloc(100 * sizeof(Scalar*));
  Scalar** visited = (Scalar**)malloc(100 * sizeof(Scalar*));

  build_topo(v, &topo, &topo_size, &visited, &visited_size);
  v->grad = 1.0;
  for (int i = topo_size - 1; i >= 0; --i) {
    topo[i]->_backward(topo[i]);
  }

  free(topo);
  free(visited);
}

void print(Scalar* a) {
  printf("Scalar[data=(%.4f), grad=(%.4f)]\n", a->data, a->grad);
}

// void backward(scalar v)
// {
//   LinkedList *visited = (LinkedList *)malloc(sizeof(LinkedList));
//   visited->head = NULL;
//   visited->tail = NULL;
//   strcpy(visited->name, "visited");
//   LinkedList *topo = (LinkedList *)malloc(sizeof(LinkedList));
//   topo->head = NULL;
//   topo->tail = NULL;
//   strcpy(topo->name, "topo");
//   build_topo(v, topo, visited);
//   free_linked_list(visited);
//   v->grad = 1.0;
//   Node *curr = topo->tail;
//   Node *temp;
//   size_t n_nodes_freed = 0;
//   while (curr)
//   {
//     if (curr->v->_backward != NULL)
//     {
//       curr->v->_backward(curr->v);
//     }
//     if (curr->v->type == TYPE_INTERMEDIATE)
//     {
//       free_scalar(curr->v);
//     }
//     temp = curr;
//     curr = curr->prev;
//     free(temp);
//   }
// }