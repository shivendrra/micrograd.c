#include "csrc/scalar.h"
#include <stdio.h>
#include <stdlib.h>

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
  print(a); // output: Scalar[data=(2), grad=0.0074]
  print(b); // output: Scalar[data=(3), grad=0.0049]
  print(c); // output: Scalar[data=(5), grad=0]
  print(d); // output: Scalar[data=(6), grad=0.0025]
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