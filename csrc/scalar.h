#ifndef Scalar_H
#define Scalar_H

typedef struct Scalar {
  double data;
  double grad;
  struct Scalar** _prev;
  int _prev_size;
  void (*_backward)(struct Scalar*);
  double aux;
} Scalar;

extern "C" {
  Scalar* initialize_Scalar(double data, Scalar** child, int child_size);
  void noop_backward(Scalar* v);

  Scalar* add_val(Scalar* a, Scalar* b);
  void add_backward(Scalar* v);
  Scalar* mul_val(Scalar* a, Scalar* b);
  void mul_backward(Scalar* v);
  Scalar* pow_val(Scalar* a, double* exp);
  void pow_backward(Scalar* v);
  
  Scalar* negate(Scalar* a);
  Scalar* sub_val(Scalar* a, Scalar* b);
  Scalar* div_val(Scalar* a, Scalar* b);
  
  Scalar* relu(Scalar* a);
  void relu_backward(Scalar* v);
  Scalar* sigmoid(Scalar* a);
  void sigmoid_backward(Scalar* v);
  Scalar* tan_h(Scalar* a);
  void tanh_backward(Scalar* v);
  Scalar* silu(Scalar* a);
  void silu_backward(Scalar* v);
  Scalar* gelu(Scalar* a);
  void gelu_backward(Scalar* v);
  Scalar* swiglu(Scalar* a);
  void swiglu_backward(Scalar* v);

  void build_topo(Scalar* v, Scalar*** topo, int* topo_size, Scalar*** visited, int* visited_size);
  void backward(Scalar* v);
  void print(Scalar* v);
}

#endif