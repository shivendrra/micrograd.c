#ifndef MODULE_H
#define MODULE_H

#include "Value.h"
#include <vector>
#include <memory>
#include <string>
#include <sstream>
#include <random>

class Module {
public:
  virtual void zero_grad();
  virtual std::vector<std::shared_ptr<Value>> parameters();
};

class Neuron : public Module {
public:
  Neuron(int nin, bool nonlin = true);
  std::shared_ptr<Value> operator()(const std::vector<std::shared_ptr<Value>>& x);
  std::vector<std::shared_ptr<Value>> parameters() override;
  std::string repr() const;

private:
  std::vector<std::shared_ptr<Value>> w;
  std::shared_ptr<Value> b;
  bool nonlin;
};

class Layer : public Module {
public:
  Layer(int nin, int nout, bool nonlin = true);
  std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& x);
  std::vector<std::shared_ptr<Value>> parameters() override;
  std::string repr() const;

private:
  std::vector<std::shared_ptr<Neuron>> neurons;
};

class MLP : public Module {
public:
  MLP(int nin, const std::vector<int>& nouts);
  std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& x);
  std::vector<std::shared_ptr<Value>> parameters() override;
  std::string repr() const;

private:
  std::vector<std::shared_ptr<Layer>> layers;
};

#endif