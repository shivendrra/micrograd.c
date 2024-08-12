#ifndef VALUE_H
#define VALUE_H

#include <vector>
#include <functional>
#include <iostream>
#include <memory>
#include <cmath>
#include <algorithm>
#include <unordered_set>

class Value {
public:
  double data;
  double grad;
  double exp;
  std::vector<std::shared_ptr<Value>> _prev;  // Changed to shared_ptr for memory management
  std::function<void(Value*)> _backward;

  Value(double data);

  static std::shared_ptr<Value> add(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b);
  static std::shared_ptr<Value> mul(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b);
  static std::shared_ptr<Value> pow_val(const std::shared_ptr<Value>& a, double exp);
  static std::shared_ptr<Value> negate(const std::shared_ptr<Value>& a);
  static std::shared_ptr<Value> sub(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b);
  static std::shared_ptr<Value> truediv(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b);
  static std::shared_ptr<Value> relu(const std::shared_ptr<Value>& a);
  static void backward(Value* v);
  void print_value() const;

private:
  static void noop_backward(Value* v);
  static void add_backward(Value* v);
  static void mul_backward(Value* v);
  static void pow_backward(Value* v);
  static void relu_backward(Value* v);
  static void build_topo(Value* v, std::vector<Value*>& topo, std::unordered_set<Value*>& visited);
};

#endif // VALUE_H
