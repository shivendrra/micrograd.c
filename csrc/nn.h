#ifndef NN_H
#define NN_H

#include "scalar.h"

typedef struct Neuron {
  tensor wei; // weights
  scalar bias;
  size_t n_in;
  int nonlin; // indicates if non-linearity is applied
} Neuron;

typedef struct Layer {
  Neuron* neurons;
  size_t n_neurons;
} Layer;

typedef struct MLP {
  Layer* layers;
  size_t n_layers;
} MLP;

Neuron* init_neuron(const size_t n_in, int nonlin);
scalar neuron_forward(Neuron* neuron, tensor inputs);

Layer* init_layer(const size_t n_in, const size_t n_out, int nonlin);
tensor layer_forward(Layer* layer, tensor inputs);

MLP* init_mlp(const size_t n_in, const size_t* n_out, size_t n_layers);
tensor mlp_forward(MLP* mlp, tensor inputs);
void mlp_free(MLP* mlp);

void zero_grad(MLP* mlp);
tensor mlp_parameters(MLP* mlp, size_t* param_count);

#endif
