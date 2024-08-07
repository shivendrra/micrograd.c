#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef struct Scalar {
  double data;
  double grad;
  void (*_backward)(struct Scalar*);
  struct Scalar** _prev;
  int _prev_size;
  float exp;
} Scalar;

void noop_backward(Scalar *v) {}

Scalar* initialize_scalars(double data, Scalar** children, int children_size) {
  Scalar* v = (Scalar*)malloc(sizeof(Scalar));
  v->data = data;
  v->grad = 0.0;
  v->_prev_size = children_size;
  v->_prev = children;
  v->_backward = noop_backward;
  v->exp = 1.0;
  return v;
}

void add_backward(Scalar* v) {
  Scalar* a = v->_prev[0];
  Scalar* b = v->_prev[1];

  a->grad += v->grad;
  b->grad += v->grad;
}

Scalar* add_val(Scalar* a, Scalar* b) {
  Scalar **children = (Scalar**)malloc(2 * sizeof(Scalar*));
  children[0] = a;
  children[1] = b;

  Scalar* out = initialize_scalars(a->data + b->data, children, 2);
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
  Scalar **children = (Scalar**)malloc(2 * sizeof(Scalar*));
  children[0] = a;
  children[1] = b;

  Scalar* out = initialize_scalars(a->data * b->data, children, 2);
  out->_backward = mul_backward;
  return out;
}

void pow_backward(Scalar* v) {
  Scalar* a = v->_prev[0];

  a->grad += v->grad * (v->exp * pow(a->data, v->exp-1));
}

Scalar* pow_val(Scalar* a, float exp) {
  Scalar **children = (Scalar**)malloc(1 * sizeof(Scalar*));
  children[0] = a;

  Scalar* out = initialize_scalars(pow(a->data, exp), children, 1);
  out->exp = exp;
  out->_backward = pow_backward;
  return out;
}

void relu_backward(Scalar* v) {
  Scalar* a = v->_prev[0];

  a->grad += v->grad * (v->data > 0);
}

Scalar* relu(Scalar* a) {
  Scalar **children = (Scalar**)malloc(1 * sizeof(Scalar*));
  children[0] = a;
  
  Scalar* out = initialize_scalars(a->data > 0 ? a->data : 0, children, 1);
  out->_backward = relu_backward;
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

int main() {
  Scalar* a = initialize_scalars(2.0, NULL, 0);
  Scalar* b = initialize_scalars(5.0, NULL, 0);

  Scalar* c = add_val(a, b);
  Scalar* d = mul_val(a, b);
  Scalar* e = pow_val(d, 3);

  backward(e);

  print(a);
  print(b);
  print(c);
  print(d);
  print(e);

  free(a);
  free(b);
  free(c);
  free(d);
  free(e);
}