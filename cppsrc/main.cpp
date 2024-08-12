#include "value.h"
#include "module.h"
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>

std::shared_ptr<Value> mse(const std::vector<std::shared_ptr<Value>>& predicted, const std::vector<std::shared_ptr<Value>>& target) {
  auto loss = std::make_shared<Value>(0.0);
  for (size_t i = 0; i < predicted.size(); ++i) {
    auto diff = Value::sub(predicted[i], target[i]);
    auto sq_diff = Value::pow_val(diff, 2.0);
    loss = Value::add(loss, sq_diff);
  }
  return Value::mul(loss, std::make_shared<Value>(1.0 / predicted.size()));
}

int main() {
  MLP mlp(2, {4, 4, 1});

  std::vector<std::shared_ptr<Value>> x = {std::make_shared<Value>(1.0), std::make_shared<Value>(2.0)};
  std::vector<std::shared_ptr<Value>> target = {std::make_shared<Value>(0.5)};

  std::cout << "Input values:\n";
  for (const auto& v : x) {
    v->print_value();
  }

  int epochs = 10;
  double learning_rate = 0.01;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    auto out = mlp(x);
    auto loss = mse(out, target);
        
    mlp.zero_grad();
        
    loss->grad = 1.0;
    Value::backward(loss.get());

    for (auto& p : mlp.parameters()) {
      p->data -= learning_rate * p->grad;
    }

    std::cout << "Epoch " << epoch << ", Loss: ";
      loss->print_value();
  }

  std::cout << "Output values after training:\n";
  auto out = mlp(x);
  for (const auto& v : out) {
    v->print_value();
  }

  std::cout << "Parameter values after training:\n";
  for (auto& p : mlp.parameters()) {
    p->print_value();
  }

  return 0;
}