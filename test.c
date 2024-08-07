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

void tanh_backward(Scalar* self) {
  Scalar* a = self->_prev[0];

  a->grad += self->grad * (1 - pow(a->data, 2));
}

Scalar* tan_h(Scalar* a) {
  Scalar** child = (Scalar**)malloc(1 * sizeof(Scalar*));
  child[0] = a;

  Scalar* out = initialize_scalars(tanh(a->data), child, 1);
  out->_backward = tanh_backward;
  return out;
}

void sigmoid_backward(Scalar* self) {
  Scalar* a = self->_prev[0];

  a->grad += self->grad * (a->data * (1 - a->data));
}

Scalar* sigmoid(Scalar* a) {
  Scalar** child = (Scalar**)malloc(1 * sizeof(Scalar*));
  child[0] = a;

  Scalar* out = initialize_scalars(1 / (1 + exp(-a->data)), child, 1);
  out->_backward = sigmoid_backward;
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
  Scalar* b = initialize_scalars(3.0, NULL, 0);

  Scalar* c = add_val(a, b);
  Scalar* d = mul_val(a, b);
  Scalar* e = relu(d);
  Scalar* f = tan_h(d);
  Scalar* g = sigmoid(d);
  Scalar* h = sub_val(a, b);

  backward(g);
  print(a); // output: Scalar[data=(2), grad=-90]
  print(b); // output: Scalar[data=(3), grad=-60]
  print(c); // output: Scalar[data=(5), grad=0]
  print(d); // output: Scalar[data=(6), grad=-30]
  print(e); // output: Scalar[data=(6), grad=0]
  print(f); // output: Scalar[data=(1), grad=0]
  print(g); // output: Scalar[data=(0.9975), grad=1]

  free(a->_prev);
  free(b->_prev);
  free(c->_prev);
  free(d->_prev);
  free(e->_prev);
  free(f->_prev);
  free(g->_prev);
  free(h->_prev);
  free(a);
  free(b);
  free(c);
  free(d);
  free(e);
  free(f);
  free(g);
  free(h);

  return 0;
}
