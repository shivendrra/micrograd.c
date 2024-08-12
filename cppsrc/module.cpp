#include "module.h"
#include <iostream>
#include <cstdlib>

void Module::zero_grad() {
  for (auto& p : parameters()) {
    p->grad = 0;
  }
}

std::vector<std::shared_ptr<Value>> Module::parameters() {
  return {};
}

Neuron::Neuron(int nin, bool nonlin) : nonlin(nonlin) {
  for (int i = 0; i < nin; ++i) {
    double random_weight = (double)rand() / RAND_MAX * 0.01; 
    w.push_back(std::make_shared<Value>(random_weight));
  }
  double random_bias = (double)rand() / RAND_MAX * 0.01;
  b = std::make_shared<Value>(random_bias);
}

std::shared_ptr<Value> Neuron::operator()(const std::vector<std::shared_ptr<Value>>& x) {
  auto sum = std::make_shared<Value>(0.0);
  for (size_t i = 0; i < x.size(); ++i) {
    sum = Value::add(sum, Value::mul(w[i], x[i]));
  }
  sum = Value::add(sum, b);
  return nonlin ? Value::relu(sum) : sum;
}

std::vector<std::shared_ptr<Value>> Neuron::parameters() {
  std::vector<std::shared_ptr<Value>> params = w;
  params.push_back(b);
  return params;
}

std::string Neuron::repr() const {
  return "Neuron";
}

Layer::Layer(int nin, int nout, bool nonlin) {
  for (int i = 0; i < nout; ++i) {
    neurons.push_back(std::make_shared<Neuron>(nin, nonlin));
  }
}

std::vector<std::shared_ptr<Value>> Layer::operator()(const std::vector<std::shared_ptr<Value>>& x) {
  std::vector<std::shared_ptr<Value>> out;
  for (auto& neuron : neurons) {
    out.push_back((*neuron)(x));
  }
  return out;
}

std::vector<std::shared_ptr<Value>> Layer::parameters() {
  std::vector<std::shared_ptr<Value>> params;
  for (auto& neuron : neurons) {
    auto neuron_params = neuron->parameters();
    params.insert(params.end(), neuron_params.begin(), neuron_params.end());
  }
  return params;
}

std::string Layer::repr() const {
  return "Layer";
}

MLP::MLP(int nin, const std::vector<int>& nouts) {
  int n = nin;
  for (auto& nout : nouts) {
    layers.push_back(std::make_shared<Layer>(n, nout));
    n = nout;
  }
}

std::vector<std::shared_ptr<Value>> MLP::operator()(const std::vector<std::shared_ptr<Value>>& x) {
  std::vector<std::shared_ptr<Value>> out = x;
  for (auto& layer : layers) {
    out = (*layer)(out);
  }
  return out;
}

std::vector<std::shared_ptr<Value>> MLP::parameters() {
  std::vector<std::shared_ptr<Value>> params;
  for (auto& layer : layers) {
    auto layer_params = layer->parameters();
    params.insert(params.end(), layer_params.begin(), layer_params.end());
  }
  return params;
}

std::string MLP::repr() const {
  return "MLP";
}