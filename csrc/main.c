#include "nn.h"
#include "scalar.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
  double xs[4][3] = {
    {2.0, 3.0, -1.0},
    {3.0, 0.0, -0.5},
    {0.5, 1.0, 1.0},
    {1.0, 1.0, -1.0}
  };
  double ys[] = {1.0, -1.0, -1.0, 1.0};

  size_t layers[] = {4, 4, 1};
  MLP* model = init_mlp(3, layers, 3);

  int epochs = 100;
  double learning_rate = 0.01;

  for (int k = 0; k < epochs; ++k) {
    scalar loss = initialize_scalars(0.0, NULL, 0);

    for (int i = 0; i < 4; ++i) {
      tensor input = (tensor)malloc(3 * sizeof(scalar));
      for (int j = 0; j < 3; ++j) {
        input[j] = initialize_scalars(xs[i][j], NULL, 0);
      }
      tensor output = mlp_forward(model, input);
      scalar target = initialize_scalars(ys[i], NULL, 0);

      scalar error = sub_val(output[0], target);
      scalar error_squared = mul_val(error, error);

      Scalar* temp = add_val(loss, error_squared);
      free(loss);
      loss = temp;

      free(input);
      free(output);
      free(target);
      free(error);
      free(error_squared);
    }

    zero_grad(model);
    backward(loss);

    size_t param_count;
    tensor params = mlp_parameters(model, &param_count);
    for (size_t i = 0; i < param_count; ++i) {
      // print(params[i]);
      params[i]->data -= learning_rate * params[i]->grad;
    }

    printf("Epoch %d -> Loss: %f\n", k, loss->data);
    free(loss);
  }

  printf("Final outputs:\n");
  for (int i = 0; i < 4; ++i) {
    tensor input = (tensor)malloc(3 * sizeof(scalar));
    for (int j = 0; j < 3; ++j) {
      input[j] = initialize_scalars(xs[i][j], NULL, 0);
    }
    tensor output = mlp_forward(model, input);
    printf("[%d] %f\n", i, output[0]->data);

    free(input);
    free(output);
  }

  mlp_free(model);

  return 0;
}