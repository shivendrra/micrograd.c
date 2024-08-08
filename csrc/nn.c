#include "nn.h"
#include "scalar.h"
#include <stdlib.h>
#include <stdio.h>

Neuron* init_neuron(const size_t n_in, int nonlin) {
  Neuron* neuron = (Neuron*)malloc(sizeof(Neuron));
  neuron->wei = (scalar*)malloc(n_in * sizeof(scalar));
  neuron->bias = initialize_scalars(0.0, NULL, 0);
  neuron->n_in = n_in;
  neuron->nonlin = nonlin;

  for (size_t i = 0; i < n_in; ++i) {
    neuron->wei[i] = initialize_scalars((double)rand() / RAND_MAX * 2 - 1, NULL, 0);
  }
  return neuron;
}

scalar neuron_forward(Neuron* neuron, tensor inputs) {
  scalar sum = initialize_scalars(neuron->bias->data, NULL, 0);
  for (size_t i = 0; i < neuron->n_in; ++i) {
    Scalar* prod = mul_val(neuron->wei[i], inputs[i]);
    Scalar* temp = add_val(sum, prod);
    free(prod);
    free(sum);
    sum = temp;
  }
  if (neuron->nonlin) {
    Scalar* temp = relu(sum);
    free(sum);
    return temp;
  }
  return sum;
}

Layer* init_layer(const size_t n_in, const size_t n_out, int nonlin) {
  Layer* layer = (Layer*)malloc(sizeof(Layer));
  layer->neurons = (Neuron*)malloc(n_out * sizeof(Neuron));
  layer->n_neurons = n_out;

  for (size_t i = 0; i < n_out; ++i) {
    layer->neurons[i] = *init_neuron(n_in, nonlin);
  }
  return layer;
}

tensor layer_forward(Layer* layer, tensor inputs) {
  tensor outputs = (tensor)malloc(layer->n_neurons * sizeof(scalar));
  for (size_t i = 0; i < layer->n_neurons; ++i) {
    outputs[i] = neuron_forward(&layer->neurons[i], inputs);
  }
  return outputs;
}

MLP* init_mlp(const size_t n_in, const size_t* n_out, size_t n_layers) {
  MLP* mlp = (MLP*)malloc(sizeof(MLP));
  mlp->layers = (Layer*)malloc(n_layers * sizeof(Layer));
  mlp->n_layers = n_layers;

  mlp->layers[0] = *init_layer(n_in, n_out[0], 1);
  for (size_t i = 1; i < n_layers; ++i) {
    mlp->layers[i] = *init_layer(n_out[i-1], n_out[i], i != n_layers - 1);
  }
  return mlp;
}

tensor mlp_forward(MLP* mlp, tensor inputs) {
  tensor output = inputs;
  for (size_t i = 0; i < mlp->n_layers; ++i) {
    tensor temp = layer_forward(&mlp->layers[i], output);
    if (i > 0) free(output);
    output = temp;
  }
  return output;
}

void mlp_free(MLP* mlp) {
  for (size_t i = 0; i < mlp->n_layers; ++i) {
    for (size_t j = 0; j < mlp->layers[i].n_neurons; ++j) {
      free(mlp->layers[i].neurons[j].wei);
      free(mlp->layers[i].neurons[j].bias);
    }
    free(mlp->layers[i].neurons);
  }
  free(mlp->layers);
  free(mlp);
}

void zero_grad(MLP* mlp) {
  for (size_t i = 0; i < mlp->n_layers; ++i) {
    for (size_t j = 0; j < mlp->layers[i].n_neurons; ++j) {
      for (size_t k = 0; k < mlp->layers[i].neurons[j].n_in; ++k) {
        mlp->layers[i].neurons[j].wei[k]->grad = 0.0;
      }
      mlp->layers[i].neurons[j].bias->grad = 0.0;
    }
  }
}

tensor mlp_parameters(MLP* mlp, size_t* param_count) {
  *param_count = 0;
  for (size_t i = 0; i < mlp->n_layers; ++i) {
    for (size_t j = 0; j < mlp->layers[i].n_neurons; ++j) {
      *param_count += mlp->layers[i].neurons[j].n_in + 1; // weights + bias
    }
  }

  tensor params = (tensor)malloc(*param_count * sizeof(scalar));
  size_t index = 0;
  for (size_t i = 0; i < mlp->n_layers; ++i) {
    for (size_t j = 0; j < mlp->layers[i].n_neurons; ++j) {
      for (size_t k = 0; k < mlp->layers[i].neurons[j].n_in; ++k) {
        params[index++] = mlp->layers[i].neurons[j].wei[k];
      }
      params[index++] = mlp->layers[i].neurons[j].bias;
    }
  }
  return params;
}
